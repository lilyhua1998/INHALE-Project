??3
??
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??,
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:*
dtype0
?
conv1d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_4/kernel
?
-conv1d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_4/kernel*"
_output_shapes
:*
dtype0
?
conv1d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv1d_transpose_4/bias

+conv1d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_4/bias*
_output_shapes
:*
dtype0
?
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma
?
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta
?
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:*
dtype0
?
conv1d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_5/kernel
?
-conv1d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_5/kernel*"
_output_shapes
:*
dtype0
?
conv1d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv1d_transpose_5/bias

+conv1d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_5/bias*
_output_shapes
:*
dtype0
?
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_17/beta
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:*
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
:*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:*
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
:*
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean
?
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance
?
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_17/moving_mean
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_17/moving_variance
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
?q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?p
value?pB?p B?p
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer-9
layer-10
layer_with_weights-6
layer-11
regularization_losses
trainable_variables
	variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
 layer_with_weights-5
 layer-8
!layer-9
"layer-10
#layer_with_weights-6
#layer-11
$regularization_losses
%trainable_variables
&	variables
'	keras_api
 
?
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
>22
?23
@24
?
(0
)1
*2
A3
B4
+5
,6
-7
.8
C9
D10
/11
012
113
214
E15
F16
317
418
519
620
G21
H22
723
824
925
:26
I27
J28
;29
<30
=31
>32
K33
L34
?35
@36
?
Mlayer_regularization_losses

Nlayers
Olayer_metrics
regularization_losses
Pnon_trainable_variables
trainable_variables
Qmetrics
	variables
 
^

(kernel
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?
Vaxis
	)gamma
*beta
Amoving_mean
Bmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
R
[regularization_losses
\trainable_variables
]	variables
^	keras_api
R
_regularization_losses
`trainable_variables
a	variables
b	keras_api
h

+kernel
,bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?
gaxis
	-gamma
.beta
Cmoving_mean
Dmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
R
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
h

/kernel
0bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?
taxis
	1gamma
2beta
Emoving_mean
Fmoving_variance
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
S
}regularization_losses
~trainable_variables
	variables
?	keras_api
b

3kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
V
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
?
(0
)1
*2
A3
B4
+5
,6
-7
.8
C9
D10
/11
012
113
214
E15
F16
317
?
 ?layer_regularization_losses
?layers
?layer_metrics
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
	variables
b

4kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	5gamma
6beta
Gmoving_mean
Hmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

7kernel
8bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	9gamma
:beta
Imoving_mean
Jmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

;kernel
<bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	=gamma
>beta
Kmoving_mean
Lmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

?kernel
@bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
^
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12
?
40
51
62
G3
H4
75
86
97
:8
I9
J10
;11
<12
=13
>14
K15
L16
?17
@18
?
 ?layer_regularization_losses
?layers
?layer_metrics
$regularization_losses
?non_trainable_variables
%trainable_variables
?metrics
&	variables
US
VARIABLE_VALUEdense_10/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_15/gamma0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_15/beta0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose_4/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv1d_transpose_4/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_16/gamma0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_16/beta0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose_5/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv1d_transpose_5/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_17/gamma0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_17/beta1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_11/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_8/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_12/gamma1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_12/beta1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d_4/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv1d_4/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_13/gamma1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_13/beta1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d_5/kernel1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv1d_5/bias1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_14/gamma1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_14/beta1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_9/kernel1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_9/bias1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_15/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_15/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_16/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_16/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_17/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_17/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_12/moving_mean'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_12/moving_variance'variables/22/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_13/moving_mean'variables/27/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_13/moving_variance'variables/28/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_14/moving_mean'variables/33/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_14/moving_variance'variables/34/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
V
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
 
 

(0

(0
?
 ?layer_regularization_losses
?layers
?layer_metrics
Rregularization_losses
?non_trainable_variables
Strainable_variables
?metrics
T	variables
 
 

)0
*1

)0
*1
A2
B3
?
 ?layer_regularization_losses
?layers
?layer_metrics
Wregularization_losses
?non_trainable_variables
Xtrainable_variables
?metrics
Y	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
[regularization_losses
?non_trainable_variables
\trainable_variables
?metrics
]	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
_regularization_losses
?non_trainable_variables
`trainable_variables
?metrics
a	variables
 

+0
,1

+0
,1
?
 ?layer_regularization_losses
?layers
?layer_metrics
cregularization_losses
?non_trainable_variables
dtrainable_variables
?metrics
e	variables
 
 

-0
.1

-0
.1
C2
D3
?
 ?layer_regularization_losses
?layers
?layer_metrics
hregularization_losses
?non_trainable_variables
itrainable_variables
?metrics
j	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
lregularization_losses
?non_trainable_variables
mtrainable_variables
?metrics
n	variables
 

/0
01

/0
01
?
 ?layer_regularization_losses
?layers
?layer_metrics
pregularization_losses
?non_trainable_variables
qtrainable_variables
?metrics
r	variables
 
 

10
21

10
21
E2
F3
?
 ?layer_regularization_losses
?layers
?layer_metrics
uregularization_losses
?non_trainable_variables
vtrainable_variables
?metrics
w	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
yregularization_losses
?non_trainable_variables
ztrainable_variables
?metrics
{	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
}regularization_losses
?non_trainable_variables
~trainable_variables
?metrics
	variables
 

30

30
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
V
0
	1

2
3
4
5
6
7
8
9
10
11
 
*
A0
B1
C2
D3
E4
F5
 
 

40

40
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 

50
61

50
61
G2
H3
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 

70
81

70
81
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 

90
:1

90
:1
I2
J3
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 

;0
<1

;0
<1
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 

=0
>1

=0
>1
K2
L3
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 

?0
@1

?0
@1
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
 
V
0
1
2
3
4
5
6
7
 8
!9
"10
#11
 
*
G0
H1
I2
J3
K4
L5
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
A0
B1
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
C0
D1
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
E0
F1
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

G0
H1
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
I0
J1
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
K0
L1
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
"serving_default_sequential_5_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_5_inputdense_10/kernel&batch_normalization_15/moving_variancebatch_normalization_15/gamma"batch_normalization_15/moving_meanbatch_normalization_15/betaconv1d_transpose_4/kernelconv1d_transpose_4/bias&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betaconv1d_transpose_5/kernelconv1d_transpose_5/bias&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betadense_11/kerneldense_8/kernel&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betaconv1d_4/kernelconv1d_4/bias&batch_normalization_13/moving_variancebatch_normalization_13/gamma"batch_normalization_13/moving_meanbatch_normalization_13/betaconv1d_5/kernelconv1d_5/bias&batch_normalization_14/moving_variancebatch_normalization_14/gamma"batch_normalization_14/moving_meanbatch_normalization_14/betadense_9/kerneldense_9/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_15809
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp-conv1d_transpose_4/kernel/Read/ReadVariableOp+conv1d_transpose_4/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp-conv1d_transpose_5/kernel/Read/ReadVariableOp+conv1d_transpose_5/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOpConst*2
Tin+
)2'*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_18303
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kernelbatch_normalization_15/gammabatch_normalization_15/betaconv1d_transpose_4/kernelconv1d_transpose_4/biasbatch_normalization_16/gammabatch_normalization_16/betaconv1d_transpose_5/kernelconv1d_transpose_5/biasbatch_normalization_17/gammabatch_normalization_17/betadense_11/kerneldense_8/kernelbatch_normalization_12/gammabatch_normalization_12/betaconv1d_4/kernelconv1d_4/biasbatch_normalization_13/gammabatch_normalization_13/betaconv1d_5/kernelconv1d_5/biasbatch_normalization_14/gammabatch_normalization_14/betadense_9/kerneldense_9/bias"batch_normalization_15/moving_mean&batch_normalization_15/moving_variance"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance"batch_normalization_12/moving_mean&batch_normalization_12/moving_variance"batch_normalization_13/moving_mean&batch_normalization_13/moving_variance"batch_normalization_14/moving_mean&batch_normalization_14/moving_variance*1
Tin*
(2&*
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
!__inference__traced_restore_18424??*
?
?
6__inference_batch_normalization_13_layer_call_fn_17836

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
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_145862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_14512

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
?
?
C__inference_conv1d_4_layer_call_and_return_conditional_losses_14535

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_13380

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
?
?
,__inference_sequential_6_layer_call_fn_15569
sequential_5_input
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*;
_read_only_resource_inputs

"#$%*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_154922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_16726

inputs+
'dense_10_matmul_readvariableop_resource0
,batch_normalization_15_assignmovingavg_165492
.batch_normalization_15_assignmovingavg_1_16555@
<batch_normalization_15_batchnorm_mul_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resourceL
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_4_biasadd_readvariableop_resource0
,batch_normalization_16_assignmovingavg_166262
.batch_normalization_16_assignmovingavg_1_16632@
<batch_normalization_16_batchnorm_mul_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resourceL
Hconv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_5_biasadd_readvariableop_resource0
,batch_normalization_17_assignmovingavg_166942
.batch_normalization_17_assignmovingavg_1_16700@
<batch_normalization_17_batchnorm_mul_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource
identity??:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_15/AssignMovingAvg/ReadVariableOp?<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_15/batchnorm/ReadVariableOp?3batch_normalization_15/batchnorm/mul/ReadVariableOp?:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_16/AssignMovingAvg/ReadVariableOp?<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_16/batchnorm/ReadVariableOp?3batch_normalization_16/batchnorm/mul/ReadVariableOp?:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_17/AssignMovingAvg/ReadVariableOp?<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_17/batchnorm/ReadVariableOp?3batch_normalization_17/batchnorm/mul/ReadVariableOp?)conv1d_transpose_4/BiasAdd/ReadVariableOp??conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_5/BiasAdd/ReadVariableOp??conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMul?
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_15/moments/mean/reduction_indices?
#batch_normalization_15/moments/meanMeandense_10/MatMul:product:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_15/moments/mean?
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_15/moments/StopGradient?
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencedense_10/MatMul:product:04batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_15/moments/SquaredDifference?
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_15/moments/variance/reduction_indices?
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_15/moments/variance?
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_15/moments/Squeeze?
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1?
,batch_normalization_15/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/16549*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_15/AssignMovingAvg/decay?
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_15_assignmovingavg_16549*
_output_shapes
:*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp?
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/16549*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/sub?
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:05batch_normalization_15/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/16549*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/mul?
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_15_assignmovingavg_16549.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/16549*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_15/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/16555*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_15/AssignMovingAvg_1/decay?
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_15_assignmovingavg_1_16555*
_output_shapes
:*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/16555*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/sub?
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:07batch_normalization_15/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/16555*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/mul?
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_15_assignmovingavg_1_165550batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/16555*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_15/batchnorm/add/y?
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add?
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt?
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp?
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul?
&batch_normalization_15/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/mul_1?
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2?
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp?
$batch_normalization_15/batchnorm/subSub7batch_normalization_15/batchnorm/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub?
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/add_1?
re_lu_6/ReluRelu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_6/Relul
reshape_5/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
reshape_5/Shape?
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_5/strided_slice/stack?
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_1?
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_2?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_5/strided_slicex
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/1x
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/2?
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_5/Reshape/shape?
reshape_5/ReshapeReshapere_lu_6/Relu:activations:0 reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_5/Reshape~
conv1d_transpose_4/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape?
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack?
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1?
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2?
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice?
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack?
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1?
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2?
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y?
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2?
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stack?
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDimsreshape_5/Reshape:output:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_4/conv1d_transpose/ExpandDims?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_slice?
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1?
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1?
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis?
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concat?
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transpose?
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze?
)conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_4/BiasAdd/ReadVariableOp?
conv1d_transpose_4/BiasAddBiasAdd4conv1d_transpose_4/conv1d_transpose/Squeeze:output:01conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_4/BiasAdd?
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_16/moments/mean/reduction_indices?
#batch_normalization_16/moments/meanMean#conv1d_transpose_4/BiasAdd:output:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_16/moments/mean?
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_16/moments/StopGradient?
0batch_normalization_16/moments/SquaredDifferenceSquaredDifference#conv1d_transpose_4/BiasAdd:output:04batch_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_16/moments/SquaredDifference?
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_16/moments/variance/reduction_indices?
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_16/moments/variance?
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_16/moments/Squeeze?
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1?
,batch_normalization_16/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/16626*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_16/AssignMovingAvg/decay?
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_16_assignmovingavg_16626*
_output_shapes
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp?
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/16626*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/sub?
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/16626*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/mul?
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_16_assignmovingavg_16626.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/16626*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_16/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/16632*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_16/AssignMovingAvg_1/decay?
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_16_assignmovingavg_1_16632*
_output_shapes
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/16632*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/sub?
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/16632*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/mul?
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_16_assignmovingavg_1_166320batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/16632*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_16/batchnorm/add/y?
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add?
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt?
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp?
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul?
&batch_normalization_16/batchnorm/mul_1Mul#conv1d_transpose_4/BiasAdd:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1?
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2?
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp?
$batch_normalization_16/batchnorm/subSub7batch_normalization_16/batchnorm/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub?
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1?
re_lu_7/ReluRelu*batch_normalization_16/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_7/Relu~
conv1d_transpose_5/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_5/Shape?
&conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_5/strided_slice/stack?
(conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice/stack_1?
(conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice/stack_2?
 conv1d_transpose_5/strided_sliceStridedSlice!conv1d_transpose_5/Shape:output:0/conv1d_transpose_5/strided_slice/stack:output:01conv1d_transpose_5/strided_slice/stack_1:output:01conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_5/strided_slice?
(conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice_1/stack?
*conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_5/strided_slice_1/stack_1?
*conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_5/strided_slice_1/stack_2?
"conv1d_transpose_5/strided_slice_1StridedSlice!conv1d_transpose_5/Shape:output:01conv1d_transpose_5/strided_slice_1/stack:output:03conv1d_transpose_5/strided_slice_1/stack_1:output:03conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_5/strided_slice_1v
conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_5/mul/y?
conv1d_transpose_5/mulMul+conv1d_transpose_5/strided_slice_1:output:0!conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_5/mulz
conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_5/stack/2?
conv1d_transpose_5/stackPack)conv1d_transpose_5/strided_slice:output:0conv1d_transpose_5/mul:z:0#conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_5/stack?
2conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_5/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDimsre_lu_7/Relu:activations:0;conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_5/conv1d_transpose/ExpandDims?
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_5/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_5/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_5/stack:output:0@conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_5/conv1d_transpose/strided_slice?
9conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_5/stack:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_5/conv1d_transpose/strided_slice_1?
3conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_5/conv1d_transpose/concat/values_1?
/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_5/conv1d_transpose/concat/axis?
*conv1d_transpose_5/conv1d_transpose/concatConcatV2:conv1d_transpose_5/conv1d_transpose/strided_slice:output:0<conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_5/conv1d_transpose/concat?
#conv1d_transpose_5/conv1d_transposeConv2DBackpropInput3conv1d_transpose_5/conv1d_transpose/concat:output:09conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_5/conv1d_transpose?
+conv1d_transpose_5/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_5/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_5/conv1d_transpose/Squeeze?
)conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_5/BiasAdd/ReadVariableOp?
conv1d_transpose_5/BiasAddBiasAdd4conv1d_transpose_5/conv1d_transpose/Squeeze:output:01conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_5/BiasAdd?
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_17/moments/mean/reduction_indices?
#batch_normalization_17/moments/meanMean#conv1d_transpose_5/BiasAdd:output:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_17/moments/mean?
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_17/moments/StopGradient?
0batch_normalization_17/moments/SquaredDifferenceSquaredDifference#conv1d_transpose_5/BiasAdd:output:04batch_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_17/moments/SquaredDifference?
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_17/moments/variance/reduction_indices?
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_17/moments/variance?
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_17/moments/Squeeze?
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_17/moments/Squeeze_1?
,batch_normalization_17/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/16694*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_17/AssignMovingAvg/decay?
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_17_assignmovingavg_16694*
_output_shapes
:*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOp?
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/16694*
_output_shapes
:2,
*batch_normalization_17/AssignMovingAvg/sub?
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/16694*
_output_shapes
:2,
*batch_normalization_17/AssignMovingAvg/mul?
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_17_assignmovingavg_16694.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/16694*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_17/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/16700*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_17/AssignMovingAvg_1/decay?
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_17_assignmovingavg_1_16700*
_output_shapes
:*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/16700*
_output_shapes
:2.
,batch_normalization_17/AssignMovingAvg_1/sub?
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/16700*
_output_shapes
:2.
,batch_normalization_17/AssignMovingAvg_1/mul?
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_17_assignmovingavg_1_167000batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/16700*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_17/batchnorm/add/y?
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/add?
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/Rsqrt?
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp?
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/mul?
&batch_normalization_17/batchnorm/mul_1Mul#conv1d_transpose_5/BiasAdd:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/mul_1?
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/mul_2?
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp?
$batch_normalization_17/batchnorm/subSub7batch_normalization_17/batchnorm/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/sub?
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/add_1?
re_lu_8/ReluRelu*batch_normalization_17/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_8/Relus
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshapere_lu_8/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshape?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_5/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanh?
IdentityIdentitydense_11/Tanh:y:0;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_15/AssignMovingAvg/ReadVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_15/batchnorm/ReadVariableOp4^batch_normalization_15/batchnorm/mul/ReadVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_16/AssignMovingAvg/ReadVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp4^batch_normalization_16/batchnorm/mul/ReadVariableOp;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_17/AssignMovingAvg/ReadVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp4^batch_normalization_17/batchnorm/mul/ReadVariableOp*^conv1d_transpose_4/BiasAdd/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_5/BiasAdd/ReadVariableOp@^conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2V
)conv1d_transpose_4/BiasAdd/ReadVariableOp)conv1d_transpose_4/BiasAdd/ReadVariableOp2?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_5/BiasAdd/ReadVariableOp)conv1d_transpose_5/BiasAdd/ReadVariableOp2?
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_10_layer_call_and_return_conditional_losses_13541

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
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_13486

inputs
assignmovingavg_13461
assignmovingavg_1_13467)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13461*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13461*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13461*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13461*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13461AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13461*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13467*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13467*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13467*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13467*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13467AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13467*
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
?<
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_13969

inputs
dense_10_13920 
batch_normalization_15_13923 
batch_normalization_15_13925 
batch_normalization_15_13927 
batch_normalization_15_13929
conv1d_transpose_4_13934
conv1d_transpose_4_13936 
batch_normalization_16_13939 
batch_normalization_16_13941 
batch_normalization_16_13943 
batch_normalization_16_13945
conv1d_transpose_5_13949
conv1d_transpose_5_13951 
batch_normalization_17_13954 
batch_normalization_17_13956 
batch_normalization_17_13958 
batch_normalization_17_13960
dense_11_13965
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?*conv1d_transpose_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_13920*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_135412"
 dense_10/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_15_13923batch_normalization_15_13925batch_normalization_15_13927batch_normalization_15_13929*
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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1313920
.batch_normalization_15/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_135932
re_lu_6/PartitionedCall?
reshape_5/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_136142
reshape_5/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_4_13934conv1d_transpose_4_13936*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_131902,
*conv1d_transpose_4/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_16_13939batch_normalization_16_13941batch_normalization_16_13943batch_normalization_16_13945*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1332920
.batch_normalization_16/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_136672
re_lu_7/PartitionedCall?
*conv1d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_transpose_5_13949conv1d_transpose_5_13951*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_133802,
*conv1d_transpose_5/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_17_13954batch_normalization_17_13956batch_normalization_17_13958batch_normalization_17_13960*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1351920
.batch_normalization_17/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_137202
re_lu_8/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_137402
flatten_5/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_11_13965*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_137562"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall+^conv1d_transpose_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2X
*conv1d_transpose_5/StatefulPartitionedCall*conv1d_transpose_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14586

inputs
assignmovingavg_14561
assignmovingavg_1_14567)
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
T0*+
_output_shapes
:?????????2
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14561*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14561*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14561*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14561*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14561AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14561*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14567*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14567*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14567*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14567*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14567AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14567*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_5_layer_call_and_return_conditional_losses_17956

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_13614

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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17459

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
?
?
#__inference_signature_wrapper_15809
sequential_5_input
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_130102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
?
?
6__inference_batch_normalization_17_layer_call_fn_17564

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_134862
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
?
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_17936

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???>2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_8_layer_call_and_return_conditional_losses_17582

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
?0
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17885

inputs
assignmovingavg_17860
assignmovingavg_1_17866)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17860*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17860*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17860*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17860*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17860AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17860*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17866*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17866*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17866*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17866*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17866AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17866*
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
?
?
C__inference_conv1d_5_layer_call_and_return_conditional_losses_14670

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15409
sequential_5_input
sequential_5_15332
sequential_5_15334
sequential_5_15336
sequential_5_15338
sequential_5_15340
sequential_5_15342
sequential_5_15344
sequential_5_15346
sequential_5_15348
sequential_5_15350
sequential_5_15352
sequential_5_15354
sequential_5_15356
sequential_5_15358
sequential_5_15360
sequential_5_15362
sequential_5_15364
sequential_5_15366
sequential_4_15369
sequential_4_15371
sequential_4_15373
sequential_4_15375
sequential_4_15377
sequential_4_15379
sequential_4_15381
sequential_4_15383
sequential_4_15385
sequential_4_15387
sequential_4_15389
sequential_4_15391
sequential_4_15393
sequential_4_15395
sequential_4_15397
sequential_4_15399
sequential_4_15401
sequential_4_15403
sequential_4_15405
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputsequential_5_15332sequential_5_15334sequential_5_15336sequential_5_15338sequential_5_15340sequential_5_15342sequential_5_15344sequential_5_15346sequential_5_15348sequential_5_15350sequential_5_15352sequential_5_15354sequential_5_15356sequential_5_15358sequential_5_15360sequential_5_15362sequential_5_15364sequential_5_15366*
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_139692&
$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0sequential_4_15369sequential_4_15371sequential_4_15373sequential_4_15375sequential_4_15377sequential_4_15379sequential_4_15381sequential_4_15383sequential_4_15385sequential_4_15387sequential_4_15389sequential_4_15391sequential_4_15393sequential_4_15395sequential_4_15397sequential_4_15399sequential_4_15401sequential_4_15403sequential_4_15405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_150392&
$sequential_4/StatefulPartitionedCall?
IdentityIdentity-sequential_4/StatefulPartitionedCall:output:0%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
?
?
6__inference_batch_normalization_14_layer_call_fn_18047

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_144172
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
?;
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_14885
dense_8_input
dense_8_14834 
batch_normalization_12_14837 
batch_normalization_12_14839 
batch_normalization_12_14841 
batch_normalization_12_14843
conv1d_4_14848
conv1d_4_14850 
batch_normalization_13_14853 
batch_normalization_13_14855 
batch_normalization_13_14857 
batch_normalization_13_14859
conv1d_5_14863
conv1d_5_14865 
batch_normalization_14_14868 
batch_normalization_14_14870 
batch_normalization_14_14872 
batch_normalization_14_14874
dense_9_14879
dense_9_14881
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_14834*
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
GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_144392!
dense_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_12_14837batch_normalization_12_14839batch_normalization_12_14841batch_normalization_12_14843*
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1413720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_144912
leaky_re_lu_6/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_145122
reshape_4/PartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv1d_4_14848conv1d_4_14850*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_145352"
 conv1d_4/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_13_14853batch_normalization_13_14855batch_normalization_13_14857batch_normalization_13_14859*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1460620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146472
leaky_re_lu_7/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv1d_5_14863conv1d_5_14865*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_146702"
 conv1d_5/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_14_14868batch_normalization_14_14870batch_normalization_14_14872batch_normalization_14_14874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1474120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_147822
leaky_re_lu_8/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_147962
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_14879dense_9_14881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_148142!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_8_input
?/
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_13106

inputs
assignmovingavg_13081
assignmovingavg_1_13087)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13081*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13081*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13081*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13081*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13081AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13081*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13087*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13087*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13087*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13087*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13087AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13087*
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
?
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_17380

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
?
^
B__inference_re_lu_6_layer_call_and_return_conditional_losses_13593

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
?0
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17803

inputs
assignmovingavg_17778
assignmovingavg_1_17784)
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
T0*+
_output_shapes
:?????????2
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17778*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17778*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17778*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17778*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17778AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17778*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17784*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17784*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17784*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17784*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17784AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17784*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_5_layer_call_fn_17604

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
GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_137402
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
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14417

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
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14137

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
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15492

inputs
sequential_5_15415
sequential_5_15417
sequential_5_15419
sequential_5_15421
sequential_5_15423
sequential_5_15425
sequential_5_15427
sequential_5_15429
sequential_5_15431
sequential_5_15433
sequential_5_15435
sequential_5_15437
sequential_5_15439
sequential_5_15441
sequential_5_15443
sequential_5_15445
sequential_5_15447
sequential_5_15449
sequential_4_15452
sequential_4_15454
sequential_4_15456
sequential_4_15458
sequential_4_15460
sequential_4_15462
sequential_4_15464
sequential_4_15466
sequential_4_15468
sequential_4_15470
sequential_4_15472
sequential_4_15474
sequential_4_15476
sequential_4_15478
sequential_4_15480
sequential_4_15482
sequential_4_15484
sequential_4_15486
sequential_4_15488
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_15415sequential_5_15417sequential_5_15419sequential_5_15421sequential_5_15423sequential_5_15425sequential_5_15427sequential_5_15429sequential_5_15431sequential_5_15433sequential_5_15435sequential_5_15437sequential_5_15439sequential_5_15441sequential_5_15443sequential_5_15445sequential_5_15447sequential_5_15449*
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_138762&
$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0sequential_4_15452sequential_4_15454sequential_4_15456sequential_4_15458sequential_4_15460sequential_4_15462sequential_4_15464sequential_4_15466sequential_4_15468sequential_4_15470sequential_4_15472sequential_4_15474sequential_4_15476sequential_4_15478sequential_4_15480sequential_4_15482sequential_4_15484sequential_4_15486sequential_4_15488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_149422&
$sequential_4/StatefulPartitionedCall?
IdentityIdentity-sequential_4/StatefulPartitionedCall:output:0%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_9_layer_call_and_return_conditional_losses_14814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
(__inference_dense_10_layer_call_fn_17293

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
C__inference_dense_10_layer_call_and_return_conditional_losses_135412
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
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_13329

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
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17349

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
?
6__inference_batch_normalization_16_layer_call_fn_17485

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_133292
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
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_13740

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
?
?
,__inference_sequential_6_layer_call_fn_16535

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity??StatefulPartitionedCall?
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_156512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_4_layer_call_and_return_conditional_losses_17758

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_15039

inputs
dense_8_14988 
batch_normalization_12_14991 
batch_normalization_12_14993 
batch_normalization_12_14995 
batch_normalization_12_14997
conv1d_4_15002
conv1d_4_15004 
batch_normalization_13_15007 
batch_normalization_13_15009 
batch_normalization_13_15011 
batch_normalization_13_15013
conv1d_5_15017
conv1d_5_15019 
batch_normalization_14_15022 
batch_normalization_14_15024 
batch_normalization_14_15026 
batch_normalization_14_15028
dense_9_15033
dense_9_15035
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_14988*
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
GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_144392!
dense_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_12_14991batch_normalization_12_14993batch_normalization_12_14995batch_normalization_12_14997*
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1413720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_144912
leaky_re_lu_6/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_145122
reshape_4/PartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv1d_4_15002conv1d_4_15004*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_145352"
 conv1d_4/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_13_15007batch_normalization_13_15009batch_normalization_13_15011batch_normalization_13_15013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1460620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146472
leaky_re_lu_7/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv1d_5_15017conv1d_5_15019*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_146702"
 conv1d_5/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_14_15022batch_normalization_14_15024batch_normalization_14_15026batch_normalization_14_15028*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1474120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_147822
leaky_re_lu_8/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_147962
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_15033dense_9_15035*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_148142!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv1d_transpose_5_layer_call_fn_13390

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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_133802
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
?
m
'__inference_dense_8_layer_call_fn_17633

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
GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_144392
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
?
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_17398

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
?
|
'__inference_dense_9_layer_call_fn_18169

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_148142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17439

inputs
assignmovingavg_17414
assignmovingavg_1_17420)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17414*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17414*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17414*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17414*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17414AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17414*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17420*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17420*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17420*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17420*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17420AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17420*
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
?
E
)__inference_flatten_4_layer_call_fn_18150

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_147962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_17490

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
?	
?
B__inference_dense_9_layer_call_and_return_conditional_losses_18160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14721

inputs
assignmovingavg_14696
assignmovingavg_1_14702)
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
T0*+
_output_shapes
:?????????2
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14696*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14696*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14696*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14696*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14696AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14696*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14702*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14702*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14702*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14702*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14702AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14702*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17823

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17531

inputs
assignmovingavg_17506
assignmovingavg_1_17512)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17506*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17506*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17506*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17506*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17506AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17506*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17512*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17512*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17512*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17512*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17512AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17512*
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
6__inference_batch_normalization_16_layer_call_fn_17472

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_132962
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
?0
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_13296

inputs
assignmovingavg_13271
assignmovingavg_1_13277)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13271*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13271*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13271*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/13271*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13271AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/13271*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13277*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13277*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13277*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/13277*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13277AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/13277*
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
?/
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17329

inputs
assignmovingavg_17304
assignmovingavg_1_17310)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17304*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17304*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17304*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17304*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17304AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17304*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17310*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17310*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17310*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17310*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17310AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17310*
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
?<
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_13876

inputs
dense_10_13827 
batch_normalization_15_13830 
batch_normalization_15_13832 
batch_normalization_15_13834 
batch_normalization_15_13836
conv1d_transpose_4_13841
conv1d_transpose_4_13843 
batch_normalization_16_13846 
batch_normalization_16_13848 
batch_normalization_16_13850 
batch_normalization_16_13852
conv1d_transpose_5_13856
conv1d_transpose_5_13858 
batch_normalization_17_13861 
batch_normalization_17_13863 
batch_normalization_17_13865 
batch_normalization_17_13867
dense_11_13872
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?*conv1d_transpose_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_13827*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_135412"
 dense_10/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_15_13830batch_normalization_15_13832batch_normalization_15_13834batch_normalization_15_13836*
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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1310620
.batch_normalization_15/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_135932
re_lu_6/PartitionedCall?
reshape_5/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_136142
reshape_5/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_4_13841conv1d_transpose_4_13843*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_131902,
*conv1d_transpose_4/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_16_13846batch_normalization_16_13848batch_normalization_16_13850batch_normalization_16_13852*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1329620
.batch_normalization_16/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_136672
re_lu_7/PartitionedCall?
*conv1d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_transpose_5_13856conv1d_transpose_5_13858*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_133802,
*conv1d_transpose_5/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_17_13861batch_normalization_17_13863batch_normalization_17_13865batch_normalization_17_13867*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1348620
.batch_normalization_17/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_137202
re_lu_8/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_137402
flatten_5/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_11_13872*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_137562"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall+^conv1d_transpose_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2X
*conv1d_transpose_5/StatefulPartitionedCall*conv1d_transpose_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_13821
dense_10_input
dense_10_13772 
batch_normalization_15_13775 
batch_normalization_15_13777 
batch_normalization_15_13779 
batch_normalization_15_13781
conv1d_transpose_4_13786
conv1d_transpose_4_13788 
batch_normalization_16_13791 
batch_normalization_16_13793 
batch_normalization_16_13795 
batch_normalization_16_13797
conv1d_transpose_5_13801
conv1d_transpose_5_13803 
batch_normalization_17_13806 
batch_normalization_17_13808 
batch_normalization_17_13810 
batch_normalization_17_13812
dense_11_13817
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?*conv1d_transpose_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_13772*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_135412"
 dense_10/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_15_13775batch_normalization_15_13777batch_normalization_15_13779batch_normalization_15_13781*
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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1313920
.batch_normalization_15/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_135932
re_lu_6/PartitionedCall?
reshape_5/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_136142
reshape_5/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_4_13786conv1d_transpose_4_13788*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_131902,
*conv1d_transpose_4/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_16_13791batch_normalization_16_13793batch_normalization_16_13795batch_normalization_16_13797*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1332920
.batch_normalization_16/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_136672
re_lu_7/PartitionedCall?
*conv1d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_transpose_5_13801conv1d_transpose_5_13803*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_133802,
*conv1d_transpose_5/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_17_13806batch_normalization_17_13808batch_normalization_17_13810batch_normalization_17_13812*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1351920
.batch_normalization_17/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_137202
re_lu_8/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_137402
flatten_5/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_11_13817*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_137562"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall+^conv1d_transpose_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2X
*conv1d_transpose_5/StatefulPartitionedCall*conv1d_transpose_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_10_input
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_13519

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
?0
?
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_13190

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
?
?
,__inference_sequential_5_layer_call_fn_14008
dense_10_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_139692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_10_input
?
E
)__inference_reshape_4_layer_call_fn_17743

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
GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_145122
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
?
?
B__inference_dense_8_layer_call_and_return_conditional_losses_14439

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
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15651

inputs
sequential_5_15574
sequential_5_15576
sequential_5_15578
sequential_5_15580
sequential_5_15582
sequential_5_15584
sequential_5_15586
sequential_5_15588
sequential_5_15590
sequential_5_15592
sequential_5_15594
sequential_5_15596
sequential_5_15598
sequential_5_15600
sequential_5_15602
sequential_5_15604
sequential_5_15606
sequential_5_15608
sequential_4_15611
sequential_4_15613
sequential_4_15615
sequential_4_15617
sequential_4_15619
sequential_4_15621
sequential_4_15623
sequential_4_15625
sequential_4_15627
sequential_4_15629
sequential_4_15631
sequential_4_15633
sequential_4_15635
sequential_4_15637
sequential_4_15639
sequential_4_15641
sequential_4_15643
sequential_4_15645
sequential_4_15647
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_15574sequential_5_15576sequential_5_15578sequential_5_15580sequential_5_15582sequential_5_15584sequential_5_15586sequential_5_15588sequential_5_15590sequential_5_15592sequential_5_15594sequential_5_15596sequential_5_15598sequential_5_15600sequential_5_15602sequential_5_15604sequential_5_15606sequential_5_15608*
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_139692&
$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0sequential_4_15611sequential_4_15613sequential_4_15615sequential_4_15617sequential_4_15619sequential_4_15621sequential_4_15623sequential_4_15625sequential_4_15627sequential_4_15629sequential_4_15631sequential_4_15633sequential_4_15635sequential_4_15637sequential_4_15639sequential_4_15641sequential_4_15643sequential_4_15645sequential_4_15647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_150392&
$sequential_4/StatefulPartitionedCall?
IdentityIdentity-sequential_4/StatefulPartitionedCall:output:0%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_13769
dense_10_input
dense_10_13550 
batch_normalization_15_13579 
batch_normalization_15_13581 
batch_normalization_15_13583 
batch_normalization_15_13585
conv1d_transpose_4_13622
conv1d_transpose_4_13624 
batch_normalization_16_13653 
batch_normalization_16_13655 
batch_normalization_16_13657 
batch_normalization_16_13659
conv1d_transpose_5_13675
conv1d_transpose_5_13677 
batch_normalization_17_13706 
batch_normalization_17_13708 
batch_normalization_17_13710 
batch_normalization_17_13712
dense_11_13765
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_4/StatefulPartitionedCall?*conv1d_transpose_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_13550*
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
C__inference_dense_10_layer_call_and_return_conditional_losses_135412"
 dense_10/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_15_13579batch_normalization_15_13581batch_normalization_15_13583batch_normalization_15_13585*
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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1310620
.batch_normalization_15/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_135932
re_lu_6/PartitionedCall?
reshape_5/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_136142
reshape_5/PartitionedCall?
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_4_13622conv1d_transpose_4_13624*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_131902,
*conv1d_transpose_4/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_16_13653batch_normalization_16_13655batch_normalization_16_13657batch_normalization_16_13659*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1329620
.batch_normalization_16/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_136672
re_lu_7/PartitionedCall?
*conv1d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_transpose_5_13675conv1d_transpose_5_13677*
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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_133802,
*conv1d_transpose_5/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_17_13706batch_normalization_17_13708batch_normalization_17_13710batch_normalization_17_13712*
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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1348620
.batch_normalization_17/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_137202
re_lu_8/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_137402
flatten_5/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_11_13765*
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
C__inference_dense_11_layer_call_and_return_conditional_losses_137562"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall+^conv1d_transpose_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2X
*conv1d_transpose_5/StatefulPartitionedCall*conv1d_transpose_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_10_input
?
?
6__inference_batch_normalization_17_layer_call_fn_17577

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_135192
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
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_18145

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?/
 __inference__wrapped_model_13010
sequential_5_inputE
Asequential_6_sequential_5_dense_10_matmul_readvariableop_resourceV
Rsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_2_resourcef
bsequential_6_sequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resourceP
Lsequential_6_sequential_5_conv1d_transpose_4_biasadd_readvariableop_resourceV
Rsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_2_resourcef
bsequential_6_sequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resourceP
Lsequential_6_sequential_5_conv1d_transpose_5_biasadd_readvariableop_resourceV
Rsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_2_resourceE
Asequential_6_sequential_5_dense_11_matmul_readvariableop_resourceD
@sequential_6_sequential_4_dense_8_matmul_readvariableop_resourceV
Rsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_2_resourceR
Nsequential_6_sequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resourceF
Bsequential_6_sequential_4_conv1d_4_biasadd_readvariableop_resourceV
Rsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_2_resourceR
Nsequential_6_sequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resourceF
Bsequential_6_sequential_4_conv1d_5_biasadd_readvariableop_resourceV
Rsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_resourceZ
Vsequential_6_sequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resourceX
Tsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_1_resourceX
Tsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_2_resourceD
@sequential_6_sequential_4_dense_9_matmul_readvariableop_resourceE
Asequential_6_sequential_4_dense_9_biasadd_readvariableop_resource
identity??Isequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2?Msequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?Isequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2?Msequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?Isequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2?Msequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?9sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOp?Esequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?9sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOp?Esequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?7sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp?8sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp?7sequential_6/sequential_4/dense_9/MatMul/ReadVariableOp?Isequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2?Msequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?Isequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2?Msequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?Isequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1?Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2?Msequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?Csequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?Ysequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?Csequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?Ysequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?8sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp?8sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp?
8sequential_6/sequential_5/dense_10/MatMul/ReadVariableOpReadVariableOpAsequential_6_sequential_5_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp?
)sequential_6/sequential_5/dense_10/MatMulMatMulsequential_5_input@sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_6/sequential_5/dense_10/MatMul?
Isequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?
@sequential_6/sequential_5/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_5/batch_normalization_15/batchnorm/add/y?
>sequential_6/sequential_5/batch_normalization_15/batchnorm/addAddV2Qsequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_5/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_15/batchnorm/add?
@sequential_6/sequential_5/batch_normalization_15/batchnorm/RsqrtRsqrtBsequential_6/sequential_5/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_15/batchnorm/Rsqrt?
Msequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_5/batch_normalization_15/batchnorm/mulMulDsequential_6/sequential_5/batch_normalization_15/batchnorm/Rsqrt:y:0Usequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_15/batchnorm/mul?
@sequential_6/sequential_5/batch_normalization_15/batchnorm/mul_1Mul3sequential_6/sequential_5/dense_10/MatMul:product:0Bsequential_6/sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_15/batchnorm/mul_1?
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_5/batch_normalization_15/batchnorm/mul_2MulSsequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_15/batchnorm/mul_2?
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_5_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_5/batch_normalization_15/batchnorm/subSubSsequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_5/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_15/batchnorm/sub?
@sequential_6/sequential_5/batch_normalization_15/batchnorm/add_1AddV2Dsequential_6/sequential_5/batch_normalization_15/batchnorm/mul_1:z:0Bsequential_6/sequential_5/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_15/batchnorm/add_1?
&sequential_6/sequential_5/re_lu_6/ReluReluDsequential_6/sequential_5/batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_6/sequential_5/re_lu_6/Relu?
)sequential_6/sequential_5/reshape_5/ShapeShape4sequential_6/sequential_5/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2+
)sequential_6/sequential_5/reshape_5/Shape?
7sequential_6/sequential_5/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_6/sequential_5/reshape_5/strided_slice/stack?
9sequential_6/sequential_5/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_6/sequential_5/reshape_5/strided_slice/stack_1?
9sequential_6/sequential_5/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_6/sequential_5/reshape_5/strided_slice/stack_2?
1sequential_6/sequential_5/reshape_5/strided_sliceStridedSlice2sequential_6/sequential_5/reshape_5/Shape:output:0@sequential_6/sequential_5/reshape_5/strided_slice/stack:output:0Bsequential_6/sequential_5/reshape_5/strided_slice/stack_1:output:0Bsequential_6/sequential_5/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_6/sequential_5/reshape_5/strided_slice?
3sequential_6/sequential_5/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_6/sequential_5/reshape_5/Reshape/shape/1?
3sequential_6/sequential_5/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_6/sequential_5/reshape_5/Reshape/shape/2?
1sequential_6/sequential_5/reshape_5/Reshape/shapePack:sequential_6/sequential_5/reshape_5/strided_slice:output:0<sequential_6/sequential_5/reshape_5/Reshape/shape/1:output:0<sequential_6/sequential_5/reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:23
1sequential_6/sequential_5/reshape_5/Reshape/shape?
+sequential_6/sequential_5/reshape_5/ReshapeReshape4sequential_6/sequential_5/re_lu_6/Relu:activations:0:sequential_6/sequential_5/reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2-
+sequential_6/sequential_5/reshape_5/Reshape?
2sequential_6/sequential_5/conv1d_transpose_4/ShapeShape4sequential_6/sequential_5/reshape_5/Reshape:output:0*
T0*
_output_shapes
:24
2sequential_6/sequential_5/conv1d_transpose_4/Shape?
@sequential_6/sequential_5/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack?
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_1?
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_2?
:sequential_6/sequential_5/conv1d_transpose_4/strided_sliceStridedSlice;sequential_6/sequential_5/conv1d_transpose_4/Shape:output:0Isequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack:output:0Ksequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_1:output:0Ksequential_6/sequential_5/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_6/sequential_5/conv1d_transpose_4/strided_slice?
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack?
Dsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_1?
Dsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_2?
<sequential_6/sequential_5/conv1d_transpose_4/strided_slice_1StridedSlice;sequential_6/sequential_5/conv1d_transpose_4/Shape:output:0Ksequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack:output:0Msequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_1:output:0Msequential_6/sequential_5/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/sequential_5/conv1d_transpose_4/strided_slice_1?
2sequential_6/sequential_5/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_6/sequential_5/conv1d_transpose_4/mul/y?
0sequential_6/sequential_5/conv1d_transpose_4/mulMulEsequential_6/sequential_5/conv1d_transpose_4/strided_slice_1:output:0;sequential_6/sequential_5/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_6/sequential_5/conv1d_transpose_4/mul?
4sequential_6/sequential_5/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_6/sequential_5/conv1d_transpose_4/stack/2?
2sequential_6/sequential_5/conv1d_transpose_4/stackPackCsequential_6/sequential_5/conv1d_transpose_4/strided_slice:output:04sequential_6/sequential_5/conv1d_transpose_4/mul:z:0=sequential_6/sequential_5/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:24
2sequential_6/sequential_5/conv1d_transpose_4/stack?
Lsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
Hsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims4sequential_6/sequential_5/reshape_5/Reshape:output:0Usequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2J
Hsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims?
Ysequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpbsequential_6_sequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02[
Ysequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Nsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2P
Nsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
Jsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsasequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Wsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2L
Jsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
Qsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
Ksequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice;sequential_6/sequential_5/conv1d_transpose_4/stack:output:0Zsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0\sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0\sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2M
Ksequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice?
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
Usequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2W
Usequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
Usequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2W
Usequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
Msequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice;sequential_6/sequential_5/conv1d_transpose_4/stack:output:0\sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0^sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0^sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2O
Msequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1?
Msequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1?
Isequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Isequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis?
Dsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concatConcatV2Tsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0Vsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0Vsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0Rsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat?
=sequential_6/sequential_5/conv1d_transpose_4/conv1d_transposeConv2DBackpropInputMsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/concat:output:0Ssequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0Qsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2?
=sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose?
Esequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/SqueezeSqueezeFsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2G
Esequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze?
Csequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpLsequential_6_sequential_5_conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?
4sequential_6/sequential_5/conv1d_transpose_4/BiasAddBiasAddNsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0Ksequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????26
4sequential_6/sequential_5/conv1d_transpose_4/BiasAdd?
Isequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?
@sequential_6/sequential_5/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_5/batch_normalization_16/batchnorm/add/y?
>sequential_6/sequential_5/batch_normalization_16/batchnorm/addAddV2Qsequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_5/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_16/batchnorm/add?
@sequential_6/sequential_5/batch_normalization_16/batchnorm/RsqrtRsqrtBsequential_6/sequential_5/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_16/batchnorm/Rsqrt?
Msequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_5/batch_normalization_16/batchnorm/mulMulDsequential_6/sequential_5/batch_normalization_16/batchnorm/Rsqrt:y:0Usequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_16/batchnorm/mul?
@sequential_6/sequential_5/batch_normalization_16/batchnorm/mul_1Mul=sequential_6/sequential_5/conv1d_transpose_4/BiasAdd:output:0Bsequential_6/sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_16/batchnorm/mul_1?
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_5/batch_normalization_16/batchnorm/mul_2MulSsequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_16/batchnorm/mul_2?
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_5_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_5/batch_normalization_16/batchnorm/subSubSsequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_5/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_16/batchnorm/sub?
@sequential_6/sequential_5/batch_normalization_16/batchnorm/add_1AddV2Dsequential_6/sequential_5/batch_normalization_16/batchnorm/mul_1:z:0Bsequential_6/sequential_5/batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_16/batchnorm/add_1?
&sequential_6/sequential_5/re_lu_7/ReluReluDsequential_6/sequential_5/batch_normalization_16/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2(
&sequential_6/sequential_5/re_lu_7/Relu?
2sequential_6/sequential_5/conv1d_transpose_5/ShapeShape4sequential_6/sequential_5/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:24
2sequential_6/sequential_5/conv1d_transpose_5/Shape?
@sequential_6/sequential_5/conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack?
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_1?
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_2?
:sequential_6/sequential_5/conv1d_transpose_5/strided_sliceStridedSlice;sequential_6/sequential_5/conv1d_transpose_5/Shape:output:0Isequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack:output:0Ksequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_1:output:0Ksequential_6/sequential_5/conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_6/sequential_5/conv1d_transpose_5/strided_slice?
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack?
Dsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_1?
Dsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_2?
<sequential_6/sequential_5/conv1d_transpose_5/strided_slice_1StridedSlice;sequential_6/sequential_5/conv1d_transpose_5/Shape:output:0Ksequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack:output:0Msequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_1:output:0Msequential_6/sequential_5/conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/sequential_5/conv1d_transpose_5/strided_slice_1?
2sequential_6/sequential_5/conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_6/sequential_5/conv1d_transpose_5/mul/y?
0sequential_6/sequential_5/conv1d_transpose_5/mulMulEsequential_6/sequential_5/conv1d_transpose_5/strided_slice_1:output:0;sequential_6/sequential_5/conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_6/sequential_5/conv1d_transpose_5/mul?
4sequential_6/sequential_5/conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_6/sequential_5/conv1d_transpose_5/stack/2?
2sequential_6/sequential_5/conv1d_transpose_5/stackPackCsequential_6/sequential_5/conv1d_transpose_5/strided_slice:output:04sequential_6/sequential_5/conv1d_transpose_5/mul:z:0=sequential_6/sequential_5/conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:24
2sequential_6/sequential_5/conv1d_transpose_5/stack?
Lsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim?
Hsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDims4sequential_6/sequential_5/re_lu_7/Relu:activations:0Usequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2J
Hsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims?
Ysequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpbsequential_6_sequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02[
Ysequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Nsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2P
Nsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim?
Jsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDimsasequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Wsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2L
Jsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1?
Qsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack?
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1?
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2?
Ksequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice;sequential_6/sequential_5/conv1d_transpose_5/stack:output:0Zsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0\sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0\sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2M
Ksequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice?
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack?
Usequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2W
Usequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1?
Usequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2W
Usequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2?
Msequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice;sequential_6/sequential_5/conv1d_transpose_5/stack:output:0\sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0^sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0^sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2O
Msequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1?
Msequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1?
Isequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Isequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis?
Dsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concatConcatV2Tsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice:output:0Vsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0Vsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:0Rsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat?
=sequential_6/sequential_5/conv1d_transpose_5/conv1d_transposeConv2DBackpropInputMsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/concat:output:0Ssequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:0Qsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2?
=sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose?
Esequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/SqueezeSqueezeFsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2G
Esequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze?
Csequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpLsequential_6_sequential_5_conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?
4sequential_6/sequential_5/conv1d_transpose_5/BiasAddBiasAddNsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze:output:0Ksequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????26
4sequential_6/sequential_5/conv1d_transpose_5/BiasAdd?
Isequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?
@sequential_6/sequential_5/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_5/batch_normalization_17/batchnorm/add/y?
>sequential_6/sequential_5/batch_normalization_17/batchnorm/addAddV2Qsequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_5/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_17/batchnorm/add?
@sequential_6/sequential_5/batch_normalization_17/batchnorm/RsqrtRsqrtBsequential_6/sequential_5/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_17/batchnorm/Rsqrt?
Msequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_5/batch_normalization_17/batchnorm/mulMulDsequential_6/sequential_5/batch_normalization_17/batchnorm/Rsqrt:y:0Usequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_17/batchnorm/mul?
@sequential_6/sequential_5/batch_normalization_17/batchnorm/mul_1Mul=sequential_6/sequential_5/conv1d_transpose_5/BiasAdd:output:0Bsequential_6/sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_17/batchnorm/mul_1?
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_5/batch_normalization_17/batchnorm/mul_2MulSsequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_5/batch_normalization_17/batchnorm/mul_2?
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_5_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_5/batch_normalization_17/batchnorm/subSubSsequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_5/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_5/batch_normalization_17/batchnorm/sub?
@sequential_6/sequential_5/batch_normalization_17/batchnorm/add_1AddV2Dsequential_6/sequential_5/batch_normalization_17/batchnorm/mul_1:z:0Bsequential_6/sequential_5/batch_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_5/batch_normalization_17/batchnorm/add_1?
&sequential_6/sequential_5/re_lu_8/ReluReluDsequential_6/sequential_5/batch_normalization_17/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2(
&sequential_6/sequential_5/re_lu_8/Relu?
)sequential_6/sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)sequential_6/sequential_5/flatten_5/Const?
+sequential_6/sequential_5/flatten_5/ReshapeReshape4sequential_6/sequential_5/re_lu_8/Relu:activations:02sequential_6/sequential_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_6/sequential_5/flatten_5/Reshape?
8sequential_6/sequential_5/dense_11/MatMul/ReadVariableOpReadVariableOpAsequential_6_sequential_5_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp?
)sequential_6/sequential_5/dense_11/MatMulMatMul4sequential_6/sequential_5/flatten_5/Reshape:output:0@sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_6/sequential_5/dense_11/MatMul?
'sequential_6/sequential_5/dense_11/TanhTanh3sequential_6/sequential_5/dense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2)
'sequential_6/sequential_5/dense_11/Tanh?
7sequential_6/sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp@sequential_6_sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp?
(sequential_6/sequential_4/dense_8/MatMulMatMul+sequential_6/sequential_5/dense_11/Tanh:y:0?sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_6/sequential_4/dense_8/MatMul?
Isequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?
@sequential_6/sequential_4/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_4/batch_normalization_12/batchnorm/add/y?
>sequential_6/sequential_4/batch_normalization_12/batchnorm/addAddV2Qsequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_4/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_12/batchnorm/add?
@sequential_6/sequential_4/batch_normalization_12/batchnorm/RsqrtRsqrtBsequential_6/sequential_4/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_12/batchnorm/Rsqrt?
Msequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_4/batch_normalization_12/batchnorm/mulMulDsequential_6/sequential_4/batch_normalization_12/batchnorm/Rsqrt:y:0Usequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_12/batchnorm/mul?
@sequential_6/sequential_4/batch_normalization_12/batchnorm/mul_1Mul2sequential_6/sequential_4/dense_8/MatMul:product:0Bsequential_6/sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_12/batchnorm/mul_1?
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_4/batch_normalization_12/batchnorm/mul_2MulSsequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_12/batchnorm/mul_2?
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_4_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_4/batch_normalization_12/batchnorm/subSubSsequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_4/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_12/batchnorm/sub?
@sequential_6/sequential_4/batch_normalization_12/batchnorm/add_1AddV2Dsequential_6/sequential_4/batch_normalization_12/batchnorm/mul_1:z:0Bsequential_6/sequential_4/batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_12/batchnorm/add_1?
1sequential_6/sequential_4/leaky_re_lu_6/LeakyRelu	LeakyReluDsequential_6/sequential_4/batch_normalization_12/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>23
1sequential_6/sequential_4/leaky_re_lu_6/LeakyRelu?
)sequential_6/sequential_4/reshape_4/ShapeShape?sequential_6/sequential_4/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2+
)sequential_6/sequential_4/reshape_4/Shape?
7sequential_6/sequential_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_6/sequential_4/reshape_4/strided_slice/stack?
9sequential_6/sequential_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_6/sequential_4/reshape_4/strided_slice/stack_1?
9sequential_6/sequential_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_6/sequential_4/reshape_4/strided_slice/stack_2?
1sequential_6/sequential_4/reshape_4/strided_sliceStridedSlice2sequential_6/sequential_4/reshape_4/Shape:output:0@sequential_6/sequential_4/reshape_4/strided_slice/stack:output:0Bsequential_6/sequential_4/reshape_4/strided_slice/stack_1:output:0Bsequential_6/sequential_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_6/sequential_4/reshape_4/strided_slice?
3sequential_6/sequential_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_6/sequential_4/reshape_4/Reshape/shape/1?
3sequential_6/sequential_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_6/sequential_4/reshape_4/Reshape/shape/2?
1sequential_6/sequential_4/reshape_4/Reshape/shapePack:sequential_6/sequential_4/reshape_4/strided_slice:output:0<sequential_6/sequential_4/reshape_4/Reshape/shape/1:output:0<sequential_6/sequential_4/reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:23
1sequential_6/sequential_4/reshape_4/Reshape/shape?
+sequential_6/sequential_4/reshape_4/ReshapeReshape?sequential_6/sequential_4/leaky_re_lu_6/LeakyRelu:activations:0:sequential_6/sequential_4/reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2-
+sequential_6/sequential_4/reshape_4/Reshape?
8sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims/dim?
4sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims
ExpandDims4sequential_6/sequential_4/reshape_4/Reshape:output:0Asequential_6/sequential_4/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????26
4sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims?
Esequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpNsequential_6_sequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02G
Esequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
:sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/dim?
6sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1
ExpandDimsMsequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0Csequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:28
6sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1?
)sequential_6/sequential_4/conv1d_4/conv1dConv2D=sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims:output:0?sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)sequential_6/sequential_4/conv1d_4/conv1d?
1sequential_6/sequential_4/conv1d_4/conv1d/SqueezeSqueeze2sequential_6/sequential_4/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????23
1sequential_6/sequential_4/conv1d_4/conv1d/Squeeze?
9sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOpReadVariableOpBsequential_6_sequential_4_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOp?
*sequential_6/sequential_4/conv1d_4/BiasAddBiasAdd:sequential_6/sequential_4/conv1d_4/conv1d/Squeeze:output:0Asequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2,
*sequential_6/sequential_4/conv1d_4/BiasAdd?
Isequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?
@sequential_6/sequential_4/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_4/batch_normalization_13/batchnorm/add/y?
>sequential_6/sequential_4/batch_normalization_13/batchnorm/addAddV2Qsequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_4/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_13/batchnorm/add?
@sequential_6/sequential_4/batch_normalization_13/batchnorm/RsqrtRsqrtBsequential_6/sequential_4/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_13/batchnorm/Rsqrt?
Msequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_4/batch_normalization_13/batchnorm/mulMulDsequential_6/sequential_4/batch_normalization_13/batchnorm/Rsqrt:y:0Usequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_13/batchnorm/mul?
@sequential_6/sequential_4/batch_normalization_13/batchnorm/mul_1Mul3sequential_6/sequential_4/conv1d_4/BiasAdd:output:0Bsequential_6/sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_13/batchnorm/mul_1?
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_4/batch_normalization_13/batchnorm/mul_2MulSsequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_13/batchnorm/mul_2?
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_4_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_4/batch_normalization_13/batchnorm/subSubSsequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_4/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_13/batchnorm/sub?
@sequential_6/sequential_4/batch_normalization_13/batchnorm/add_1AddV2Dsequential_6/sequential_4/batch_normalization_13/batchnorm/mul_1:z:0Bsequential_6/sequential_4/batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_13/batchnorm/add_1?
1sequential_6/sequential_4/leaky_re_lu_7/LeakyRelu	LeakyReluDsequential_6/sequential_4/batch_normalization_13/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>23
1sequential_6/sequential_4/leaky_re_lu_7/LeakyRelu?
8sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims/dim?
4sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims
ExpandDims?sequential_6/sequential_4/leaky_re_lu_7/LeakyRelu:activations:0Asequential_6/sequential_4/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????26
4sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims?
Esequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpNsequential_6_sequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02G
Esequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
:sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/dim?
6sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1
ExpandDimsMsequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0Csequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:28
6sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1?
)sequential_6/sequential_4/conv1d_5/conv1dConv2D=sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims:output:0?sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)sequential_6/sequential_4/conv1d_5/conv1d?
1sequential_6/sequential_4/conv1d_5/conv1d/SqueezeSqueeze2sequential_6/sequential_4/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????23
1sequential_6/sequential_4/conv1d_5/conv1d/Squeeze?
9sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOpReadVariableOpBsequential_6_sequential_4_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOp?
*sequential_6/sequential_4/conv1d_5/BiasAddBiasAdd:sequential_6/sequential_4/conv1d_5/conv1d/Squeeze:output:0Asequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2,
*sequential_6/sequential_4/conv1d_5/BiasAdd?
Isequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOpRsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?
@sequential_6/sequential_4/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_6/sequential_4/batch_normalization_14/batchnorm/add/y?
>sequential_6/sequential_4/batch_normalization_14/batchnorm/addAddV2Qsequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp:value:0Isequential_6/sequential_4/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_14/batchnorm/add?
@sequential_6/sequential_4/batch_normalization_14/batchnorm/RsqrtRsqrtBsequential_6/sequential_4/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_14/batchnorm/Rsqrt?
Msequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_6_sequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?
>sequential_6/sequential_4/batch_normalization_14/batchnorm/mulMulDsequential_6/sequential_4/batch_normalization_14/batchnorm/Rsqrt:y:0Usequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_14/batchnorm/mul?
@sequential_6/sequential_4/batch_normalization_14/batchnorm/mul_1Mul3sequential_6/sequential_4/conv1d_5/BiasAdd:output:0Bsequential_6/sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_14/batchnorm/mul_1?
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?
@sequential_6/sequential_4/batch_normalization_14/batchnorm/mul_2MulSsequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1:value:0Bsequential_6/sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_6/sequential_4/batch_normalization_14/batchnorm/mul_2?
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_6_sequential_4_batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2?
>sequential_6/sequential_4/batch_normalization_14/batchnorm/subSubSsequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2:value:0Dsequential_6/sequential_4/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_6/sequential_4/batch_normalization_14/batchnorm/sub?
@sequential_6/sequential_4/batch_normalization_14/batchnorm/add_1AddV2Dsequential_6/sequential_4/batch_normalization_14/batchnorm/mul_1:z:0Bsequential_6/sequential_4/batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_6/sequential_4/batch_normalization_14/batchnorm/add_1?
1sequential_6/sequential_4/leaky_re_lu_8/LeakyRelu	LeakyReluDsequential_6/sequential_4/batch_normalization_14/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>23
1sequential_6/sequential_4/leaky_re_lu_8/LeakyRelu?
)sequential_6/sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)sequential_6/sequential_4/flatten_4/Const?
+sequential_6/sequential_4/flatten_4/ReshapeReshape?sequential_6/sequential_4/leaky_re_lu_8/LeakyRelu:activations:02sequential_6/sequential_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_6/sequential_4/flatten_4/Reshape?
7sequential_6/sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp@sequential_6_sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_6/sequential_4/dense_9/MatMul/ReadVariableOp?
(sequential_6/sequential_4/dense_9/MatMulMatMul4sequential_6/sequential_4/flatten_4/Reshape:output:0?sequential_6/sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_6/sequential_4/dense_9/MatMul?
8sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp?
)sequential_6/sequential_4/dense_9/BiasAddBiasAdd2sequential_6/sequential_4/dense_9/MatMul:product:0@sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_6/sequential_4/dense_9/BiasAdd?
IdentityIdentity2sequential_6/sequential_4/dense_9/BiasAdd:output:0J^sequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOpL^sequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1L^sequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2N^sequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpJ^sequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOpL^sequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1L^sequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2N^sequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpJ^sequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOpL^sequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1L^sequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2N^sequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp:^sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOpF^sequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:^sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOpF^sequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp8^sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp9^sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp8^sequential_6/sequential_4/dense_9/MatMul/ReadVariableOpJ^sequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOpL^sequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1L^sequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2N^sequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpJ^sequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOpL^sequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1L^sequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2N^sequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpJ^sequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOpL^sequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1L^sequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2N^sequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOpD^sequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpZ^sequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpD^sequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpZ^sequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp9^sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp9^sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2?
Isequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOpIsequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2Ksequential_6/sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpMsequential_6/sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp2?
Isequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOpIsequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2Ksequential_6/sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpMsequential_6/sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp2?
Isequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOpIsequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2Ksequential_6/sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOpMsequential_6/sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp2v
9sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOp9sequential_6/sequential_4/conv1d_4/BiasAdd/ReadVariableOp2?
Esequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpEsequential_6/sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2v
9sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOp9sequential_6/sequential_4/conv1d_5/BiasAdd/ReadVariableOp2?
Esequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpEsequential_6/sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2r
7sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp7sequential_6/sequential_4/dense_8/MatMul/ReadVariableOp2t
8sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp8sequential_6/sequential_4/dense_9/BiasAdd/ReadVariableOp2r
7sequential_6/sequential_4/dense_9/MatMul/ReadVariableOp7sequential_6/sequential_4/dense_9/MatMul/ReadVariableOp2?
Isequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOpIsequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2Ksequential_6/sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpMsequential_6/sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp2?
Isequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOpIsequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2Ksequential_6/sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpMsequential_6/sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp2?
Isequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOpIsequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp2?
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_12?
Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2Ksequential_6/sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_22?
Msequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOpMsequential_6/sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp2?
Csequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpCsequential_6/sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp2?
Ysequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpYsequential_6/sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
Csequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpCsequential_6/sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp2?
Ysequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpYsequential_6/sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2t
8sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp8sequential_6/sequential_5/dense_10/MatMul/ReadVariableOp2t
8sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp8sequential_6/sequential_5/dense_11/MatMul/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
?
C
'__inference_re_lu_7_layer_call_fn_17495

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
GPU 2J 8? *K
fFRD
B__inference_re_lu_7_layer_call_and_return_conditional_losses_136672
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
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18021

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
??
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_17096

inputs*
&dense_8_matmul_readvariableop_resource0
,batch_normalization_12_assignmovingavg_169652
.batch_normalization_12_assignmovingavg_1_16971@
<batch_normalization_12_batchnorm_mul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource0
,batch_normalization_13_assignmovingavg_170182
.batch_normalization_13_assignmovingavg_1_17024@
<batch_normalization_13_batchnorm_mul_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource0
,batch_normalization_14_assignmovingavg_170622
.batch_normalization_14_assignmovingavg_1_17068@
<batch_normalization_14_batchnorm_mul_readvariableop_resource<
8batch_normalization_14_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_12/AssignMovingAvg/ReadVariableOp?<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_12/batchnorm/ReadVariableOp?3batch_normalization_12/batchnorm/mul/ReadVariableOp?:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_13/AssignMovingAvg/ReadVariableOp?<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_13/batchnorm/ReadVariableOp?3batch_normalization_13/batchnorm/mul/ReadVariableOp?:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_14/AssignMovingAvg/ReadVariableOp?<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_14/batchnorm/ReadVariableOp?3batch_normalization_14/batchnorm/mul/ReadVariableOp?conv1d_4/BiasAdd/ReadVariableOp?+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?conv1d_5/BiasAdd/ReadVariableOp?+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_12/moments/mean/reduction_indices?
#batch_normalization_12/moments/meanMeandense_8/MatMul:product:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_12/moments/mean?
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_12/moments/StopGradient?
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencedense_8/MatMul:product:04batch_normalization_12/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_12/moments/SquaredDifference?
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_12/moments/variance/reduction_indices?
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_12/moments/variance?
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_12/moments/Squeeze?
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1?
,batch_normalization_12/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_12/AssignMovingAvg/16965*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_12/AssignMovingAvg/decay?
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_12_assignmovingavg_16965*
_output_shapes
:*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOp?
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_12/AssignMovingAvg/16965*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/sub?
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_12/AssignMovingAvg/16965*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/mul?
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_12_assignmovingavg_16965.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_12/AssignMovingAvg/16965*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_12/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg_1/16971*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_12/AssignMovingAvg_1/decay?
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_12_assignmovingavg_1_16971*
_output_shapes
:*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg_1/16971*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/sub?
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg_1/16971*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/mul?
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_12_assignmovingavg_1_169710batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg_1/16971*
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
:2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/mul_1?
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp?
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/add_1?
leaky_re_lu_6/LeakyRelu	LeakyRelu*batch_normalization_12/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_6/LeakyReluw
reshape_4/ShapeShape%leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape%leaky_re_lu_6/LeakyRelu:activations:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_4/Reshape?
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_4/conv1d/ExpandDims/dim?
conv1d_4/conv1d/ExpandDims
ExpandDimsreshape_4/Reshape:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_4/conv1d/ExpandDims?
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim?
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_4/conv1d/ExpandDims_1?
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_4/conv1d?
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_4/conv1d/Squeeze?
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp?
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_4/BiasAdd?
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_13/moments/mean/reduction_indices?
#batch_normalization_13/moments/meanMeanconv1d_4/BiasAdd:output:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_13/moments/mean?
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_13/moments/StopGradient?
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferenceconv1d_4/BiasAdd:output:04batch_normalization_13/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_13/moments/SquaredDifference?
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_13/moments/variance/reduction_indices?
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_13/moments/variance?
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_13/moments/Squeeze?
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1?
,batch_normalization_13/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_13/AssignMovingAvg/17018*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_13/AssignMovingAvg/decay?
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_13_assignmovingavg_17018*
_output_shapes
:*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOp?
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_13/AssignMovingAvg/17018*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/sub?
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_13/AssignMovingAvg/17018*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/mul?
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_13_assignmovingavg_17018.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_13/AssignMovingAvg/17018*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_13/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg_1/17024*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_13/AssignMovingAvg_1/decay?
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_13_assignmovingavg_1_17024*
_output_shapes
:*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg_1/17024*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/sub?
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg_1/17024*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/mul?
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_13_assignmovingavg_1_170240batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg_1/17024*
_output_shapes
 *
dtype02>
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_13/batchnorm/add/y?
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add?
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt?
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOp?
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul?
&batch_normalization_13/batchnorm/mul_1Mulconv1d_4/BiasAdd:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_13/batchnorm/mul_1?
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2?
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOp?
$batch_normalization_13/batchnorm/subSub7batch_normalization_13/batchnorm/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub?
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_13/batchnorm/add_1?
leaky_re_lu_7/LeakyRelu	LeakyRelu*batch_normalization_13/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_7/LeakyRelu?
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_5/conv1d/ExpandDims/dim?
conv1d_5/conv1d/ExpandDims
ExpandDims%leaky_re_lu_7/LeakyRelu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_5/conv1d/ExpandDims?
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim?
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_5/conv1d/ExpandDims_1?
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_5/conv1d?
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_5/conv1d/Squeeze?
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp?
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_5/BiasAdd?
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_14/moments/mean/reduction_indices?
#batch_normalization_14/moments/meanMeanconv1d_5/BiasAdd:output:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_14/moments/mean?
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_14/moments/StopGradient?
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferenceconv1d_5/BiasAdd:output:04batch_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_14/moments/SquaredDifference?
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_14/moments/variance/reduction_indices?
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_14/moments/variance?
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_14/moments/Squeeze?
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_14/moments/Squeeze_1?
,batch_normalization_14/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_14/AssignMovingAvg/17062*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_14/AssignMovingAvg/decay?
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_14_assignmovingavg_17062*
_output_shapes
:*
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOp?
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_14/AssignMovingAvg/17062*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/sub?
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:05batch_normalization_14/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_14/AssignMovingAvg/17062*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/mul?
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_14_assignmovingavg_17062.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_14/AssignMovingAvg/17062*
_output_shapes
 *
dtype02<
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_14/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_14/AssignMovingAvg_1/17068*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_14/AssignMovingAvg_1/decay?
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_14_assignmovingavg_1_17068*
_output_shapes
:*
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_14/AssignMovingAvg_1/17068*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/sub?
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:07batch_normalization_14/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_14/AssignMovingAvg_1/17068*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/mul?
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_14_assignmovingavg_1_170680batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_14/AssignMovingAvg_1/17068*
_output_shapes
 *
dtype02>
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_14/batchnorm/add/y?
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add?
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt?
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_14/batchnorm/mul/ReadVariableOp?
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul?
&batch_normalization_14/batchnorm/mul_1Mulconv1d_5/BiasAdd:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/mul_1?
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2?
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_14/batchnorm/ReadVariableOp?
$batch_normalization_14/batchnorm/subSub7batch_normalization_14/batchnorm/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub?
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/add_1?
leaky_re_lu_8/LeakyRelu	LeakyRelu*batch_normalization_14/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_8/LeakyRelus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshape%leaky_re_lu_8/LeakyRelu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd?

IdentityIdentitydense_9/BiasAdd:output:0;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_12/AssignMovingAvg/ReadVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_13/AssignMovingAvg/ReadVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp4^batch_normalization_13/batchnorm/mul/ReadVariableOp;^batch_normalization_14/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_14/AssignMovingAvg/ReadVariableOp=^batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_14/batchnorm/ReadVariableOp4^batch_normalization_14/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2x
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
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2x
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp:batch_normalization_14/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_14/AssignMovingAvg/ReadVariableOp5batch_normalization_14/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_14/batchnorm/ReadVariableOp/batch_normalization_14/batchnorm/ReadVariableOp2j
3batch_normalization_14/batchnorm/mul/ReadVariableOp3batch_normalization_14/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_13915
dense_10_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_138762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_10_input
?
I
-__inference_leaky_re_lu_8_layer_call_fn_18139

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_147822
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?N
?
__inference__traced_save_18303
file_prefix.
*savev2_dense_10_kernel_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableop8
4savev2_conv1d_transpose_4_kernel_read_readvariableop6
2savev2_conv1d_transpose_4_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableop8
4savev2_conv1d_transpose_5_kernel_read_readvariableop6
2savev2_conv1d_transpose_5_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop4savev2_conv1d_transpose_4_kernel_read_readvariableop2savev2_conv1d_transpose_4_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop4savev2_conv1d_transpose_5_kernel_read_readvariableop2savev2_conv1d_transpose_5_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop*savev2_dense_11_kernel_read_readvariableop)savev2_dense_8_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::: 2(
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
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
::&

_output_shapes
: 
?
?
6__inference_batch_normalization_13_layer_call_fn_17931

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_142772
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
?
C
'__inference_re_lu_8_layer_call_fn_17587

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
GPU 2J 8? *K
fFRD
B__inference_re_lu_8_layer_call_and_return_conditional_losses_137202
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
?:
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_14942

inputs
dense_8_14891 
batch_normalization_12_14894 
batch_normalization_12_14896 
batch_normalization_12_14898 
batch_normalization_12_14900
conv1d_4_14905
conv1d_4_14907 
batch_normalization_13_14910 
batch_normalization_13_14912 
batch_normalization_13_14914 
batch_normalization_13_14916
conv1d_5_14920
conv1d_5_14922 
batch_normalization_14_14925 
batch_normalization_14_14927 
batch_normalization_14_14929 
batch_normalization_14_14931
dense_9_14936
dense_9_14938
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_14891*
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
GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_144392!
dense_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_12_14894batch_normalization_12_14896batch_normalization_12_14898batch_normalization_12_14900*
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1410420
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_144912
leaky_re_lu_6/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_145122
reshape_4/PartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv1d_4_14905conv1d_4_14907*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_145352"
 conv1d_4/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_13_14910batch_normalization_13_14912batch_normalization_13_14914batch_normalization_13_14916*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1458620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146472
leaky_re_lu_7/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv1d_5_14920conv1d_5_14922*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_146702"
 conv1d_5/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_14_14925batch_normalization_14_14927batch_normalization_14_14929batch_normalization_14_14931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1472120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_147822
leaky_re_lu_8/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_147962
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_14936dense_9_14938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_148142!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_6_layer_call_fn_16456

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity??StatefulPartitionedCall?
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*;
_read_only_resource_inputs

"#$%*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_154922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_14796

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_4_layer_call_fn_14983
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

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_149422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_8_input
?
?
B__inference_dense_8_layer_call_and_return_conditional_losses_17626

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
?0
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14384

inputs
assignmovingavg_14359
assignmovingavg_1_14365)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14359*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14359*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14359*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14359*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14359AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14359*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14365*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14365*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14365*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14365*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14365AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14365*
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
?/
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17669

inputs
assignmovingavg_17644
assignmovingavg_1_17650)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17644*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17644*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17644*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17644*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17644AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17644*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17650*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17650*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17650*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17650*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17650AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17650*
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
?;
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_14831
dense_8_input
dense_8_14448 
batch_normalization_12_14477 
batch_normalization_12_14479 
batch_normalization_12_14481 
batch_normalization_12_14483
conv1d_4_14546
conv1d_4_14548 
batch_normalization_13_14633 
batch_normalization_13_14635 
batch_normalization_13_14637 
batch_normalization_13_14639
conv1d_5_14681
conv1d_5_14683 
batch_normalization_14_14768 
batch_normalization_14_14770 
batch_normalization_14_14772 
batch_normalization_14_14774
dense_9_14825
dense_9_14827
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_14448*
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
GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_144392!
dense_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_12_14477batch_normalization_12_14479batch_normalization_12_14481batch_normalization_12_14483*
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1410420
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_144912
leaky_re_lu_6/PartitionedCall?
reshape_4/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_145122
reshape_4/PartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv1d_4_14546conv1d_4_14548*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_145352"
 conv1d_4/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_13_14633batch_normalization_13_14635batch_normalization_13_14637batch_normalization_13_14639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1458620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146472
leaky_re_lu_7/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv1d_5_14681conv1d_5_14683*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_146702"
 conv1d_5/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_14_14768batch_normalization_14_14770batch_normalization_14_14772batch_normalization_14_14774*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1472120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_147822
leaky_re_lu_8/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_147962
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_14825dense_9_14827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_148142!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_8_input
?
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_17720

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
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
?
6__inference_batch_normalization_13_layer_call_fn_17849

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
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_146062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?.
G__inference_sequential_6_layer_call_and_return_conditional_losses_16141

inputs8
4sequential_5_dense_10_matmul_readvariableop_resource=
9sequential_5_batch_normalization_15_assignmovingavg_15823?
;sequential_5_batch_normalization_15_assignmovingavg_1_15829M
Isequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resourceI
Esequential_5_batch_normalization_15_batchnorm_readvariableop_resourceY
Usequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_5_conv1d_transpose_4_biasadd_readvariableop_resource=
9sequential_5_batch_normalization_16_assignmovingavg_15900?
;sequential_5_batch_normalization_16_assignmovingavg_1_15906M
Isequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resourceI
Esequential_5_batch_normalization_16_batchnorm_readvariableop_resourceY
Usequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_5_conv1d_transpose_5_biasadd_readvariableop_resource=
9sequential_5_batch_normalization_17_assignmovingavg_15968?
;sequential_5_batch_normalization_17_assignmovingavg_1_15974M
Isequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resourceI
Esequential_5_batch_normalization_17_batchnorm_readvariableop_resource8
4sequential_5_dense_11_matmul_readvariableop_resource7
3sequential_4_dense_8_matmul_readvariableop_resource=
9sequential_4_batch_normalization_12_assignmovingavg_16010?
;sequential_4_batch_normalization_12_assignmovingavg_1_16016M
Isequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resourceI
Esequential_4_batch_normalization_12_batchnorm_readvariableop_resourceE
Asequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resource9
5sequential_4_conv1d_4_biasadd_readvariableop_resource=
9sequential_4_batch_normalization_13_assignmovingavg_16063?
;sequential_4_batch_normalization_13_assignmovingavg_1_16069M
Isequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resourceI
Esequential_4_batch_normalization_13_batchnorm_readvariableop_resourceE
Asequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resource9
5sequential_4_conv1d_5_biasadd_readvariableop_resource=
9sequential_4_batch_normalization_14_assignmovingavg_16107?
;sequential_4_batch_normalization_14_assignmovingavg_1_16113M
Isequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resourceI
Esequential_4_batch_normalization_14_batchnorm_readvariableop_resource7
3sequential_4_dense_9_matmul_readvariableop_resource8
4sequential_4_dense_9_biasadd_readvariableop_resource
identity??Gsequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?Bsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOp?Isequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?Gsequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOp?Bsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOp?Isequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp?<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?Gsequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOp?Bsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOp?Isequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp?<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?,sequential_4/conv1d_4/BiasAdd/ReadVariableOp?8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?,sequential_4/conv1d_5/BiasAdd/ReadVariableOp?8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?*sequential_4/dense_8/MatMul/ReadVariableOp?+sequential_4/dense_9/BiasAdd/ReadVariableOp?*sequential_4/dense_9/MatMul/ReadVariableOp?Gsequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?Bsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOp?Isequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?Gsequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?Bsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOp?Isequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?Gsequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?Bsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOp?Isequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?+sequential_5/dense_10/MatMul/ReadVariableOp?+sequential_5/dense_11/MatMul/ReadVariableOp?
+sequential_5/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_5/dense_10/MatMul/ReadVariableOp?
sequential_5/dense_10/MatMulMatMulinputs3sequential_5/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_10/MatMul?
Bsequential_5/batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_5/batch_normalization_15/moments/mean/reduction_indices?
0sequential_5/batch_normalization_15/moments/meanMean&sequential_5/dense_10/MatMul:product:0Ksequential_5/batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(22
0sequential_5/batch_normalization_15/moments/mean?
8sequential_5/batch_normalization_15/moments/StopGradientStopGradient9sequential_5/batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:2:
8sequential_5/batch_normalization_15/moments/StopGradient?
=sequential_5/batch_normalization_15/moments/SquaredDifferenceSquaredDifference&sequential_5/dense_10/MatMul:product:0Asequential_5/batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2?
=sequential_5/batch_normalization_15/moments/SquaredDifference?
Fsequential_5/batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_5/batch_normalization_15/moments/variance/reduction_indices?
4sequential_5/batch_normalization_15/moments/varianceMeanAsequential_5/batch_normalization_15/moments/SquaredDifference:z:0Osequential_5/batch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(26
4sequential_5/batch_normalization_15/moments/variance?
3sequential_5/batch_normalization_15/moments/SqueezeSqueeze9sequential_5/batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_5/batch_normalization_15/moments/Squeeze?
5sequential_5/batch_normalization_15/moments/Squeeze_1Squeeze=sequential_5/batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_5/batch_normalization_15/moments/Squeeze_1?
9sequential_5/batch_normalization_15/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_15/AssignMovingAvg/15823*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_5/batch_normalization_15/AssignMovingAvg/decay?
Bsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_5_batch_normalization_15_assignmovingavg_15823*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOp?
7sequential_5/batch_normalization_15/AssignMovingAvg/subSubJsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0<sequential_5/batch_normalization_15/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_15/AssignMovingAvg/15823*
_output_shapes
:29
7sequential_5/batch_normalization_15/AssignMovingAvg/sub?
7sequential_5/batch_normalization_15/AssignMovingAvg/mulMul;sequential_5/batch_normalization_15/AssignMovingAvg/sub:z:0Bsequential_5/batch_normalization_15/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_15/AssignMovingAvg/15823*
_output_shapes
:29
7sequential_5/batch_normalization_15/AssignMovingAvg/mul?
Gsequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_5_batch_normalization_15_assignmovingavg_15823;sequential_5/batch_normalization_15/AssignMovingAvg/mul:z:0C^sequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_15/AssignMovingAvg/15823*
_output_shapes
 *
dtype02I
Gsequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?
;sequential_5/batch_normalization_15/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_15/AssignMovingAvg_1/15829*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_5/batch_normalization_15/AssignMovingAvg_1/decay?
Dsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_15_assignmovingavg_1_15829*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?
9sequential_5/batch_normalization_15/AssignMovingAvg_1/subSubLsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_5/batch_normalization_15/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_15/AssignMovingAvg_1/15829*
_output_shapes
:2;
9sequential_5/batch_normalization_15/AssignMovingAvg_1/sub?
9sequential_5/batch_normalization_15/AssignMovingAvg_1/mulMul=sequential_5/batch_normalization_15/AssignMovingAvg_1/sub:z:0Dsequential_5/batch_normalization_15/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_15/AssignMovingAvg_1/15829*
_output_shapes
:2;
9sequential_5/batch_normalization_15/AssignMovingAvg_1/mul?
Isequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_5_batch_normalization_15_assignmovingavg_1_15829=sequential_5/batch_normalization_15/AssignMovingAvg_1/mul:z:0E^sequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_15/AssignMovingAvg_1/15829*
_output_shapes
 *
dtype02K
Isequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_5/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_15/batchnorm/add/y?
1sequential_5/batch_normalization_15/batchnorm/addAddV2>sequential_5/batch_normalization_15/moments/Squeeze_1:output:0<sequential_5/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/add?
3sequential_5/batch_normalization_15/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_15/batchnorm/Rsqrt?
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_15/batchnorm/mulMul7sequential_5/batch_normalization_15/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/mul?
3sequential_5/batch_normalization_15/batchnorm/mul_1Mul&sequential_5/dense_10/MatMul:product:05sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????25
3sequential_5/batch_normalization_15/batchnorm/mul_1?
3sequential_5/batch_normalization_15/batchnorm/mul_2Mul<sequential_5/batch_normalization_15/moments/Squeeze:output:05sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_15/batchnorm/mul_2?
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?
1sequential_5/batch_normalization_15/batchnorm/subSubDsequential_5/batch_normalization_15/batchnorm/ReadVariableOp:value:07sequential_5/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/sub?
3sequential_5/batch_normalization_15/batchnorm/add_1AddV27sequential_5/batch_normalization_15/batchnorm/mul_1:z:05sequential_5/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????25
3sequential_5/batch_normalization_15/batchnorm/add_1?
sequential_5/re_lu_6/ReluRelu7sequential_5/batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
sequential_5/re_lu_6/Relu?
sequential_5/reshape_5/ShapeShape'sequential_5/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
sequential_5/reshape_5/Shape?
*sequential_5/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/reshape_5/strided_slice/stack?
,sequential_5/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_5/strided_slice/stack_1?
,sequential_5/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_5/strided_slice/stack_2?
$sequential_5/reshape_5/strided_sliceStridedSlice%sequential_5/reshape_5/Shape:output:03sequential_5/reshape_5/strided_slice/stack:output:05sequential_5/reshape_5/strided_slice/stack_1:output:05sequential_5/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/reshape_5/strided_slice?
&sequential_5/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_5/Reshape/shape/1?
&sequential_5/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_5/Reshape/shape/2?
$sequential_5/reshape_5/Reshape/shapePack-sequential_5/reshape_5/strided_slice:output:0/sequential_5/reshape_5/Reshape/shape/1:output:0/sequential_5/reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_5/reshape_5/Reshape/shape?
sequential_5/reshape_5/ReshapeReshape'sequential_5/re_lu_6/Relu:activations:0-sequential_5/reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_5/reshape_5/Reshape?
%sequential_5/conv1d_transpose_4/ShapeShape'sequential_5/reshape_5/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_4/Shape?
3sequential_5/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv1d_transpose_4/strided_slice/stack?
5sequential_5/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice/stack_1?
5sequential_5/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice/stack_2?
-sequential_5/conv1d_transpose_4/strided_sliceStridedSlice.sequential_5/conv1d_transpose_4/Shape:output:0<sequential_5/conv1d_transpose_4/strided_slice/stack:output:0>sequential_5/conv1d_transpose_4/strided_slice/stack_1:output:0>sequential_5/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv1d_transpose_4/strided_slice?
5sequential_5/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice_1/stack?
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_1?
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_2?
/sequential_5/conv1d_transpose_4/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_4/Shape:output:0>sequential_5/conv1d_transpose_4/strided_slice_1/stack:output:0@sequential_5/conv1d_transpose_4/strided_slice_1/stack_1:output:0@sequential_5/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv1d_transpose_4/strided_slice_1?
%sequential_5/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/conv1d_transpose_4/mul/y?
#sequential_5/conv1d_transpose_4/mulMul8sequential_5/conv1d_transpose_4/strided_slice_1:output:0.sequential_5/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_5/conv1d_transpose_4/mul?
'sequential_5/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv1d_transpose_4/stack/2?
%sequential_5/conv1d_transpose_4/stackPack6sequential_5/conv1d_transpose_4/strided_slice:output:0'sequential_5/conv1d_transpose_4/mul:z:00sequential_5/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_4/stack?
?sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
;sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims'sequential_5/reshape_5/Reshape:output:0Hsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims?
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
=sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
Dsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
>sequential_5/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice.sequential_5/conv1d_transpose_4/stack:output:0Msequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
@sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_4/stack:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1?
@sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1?
<sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis?
7sequential_5/conv1d_transpose_4/conv1d_transpose/concatConcatV2Gsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0Isequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0Isequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0Esequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_5/conv1d_transpose_4/conv1d_transpose/concat?
0sequential_5/conv1d_transpose_4/conv1d_transposeConv2DBackpropInput@sequential_5/conv1d_transpose_4/conv1d_transpose/concat:output:0Fsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0Dsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_5/conv1d_transpose_4/conv1d_transpose?
8sequential_5/conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze9sequential_5/conv1d_transpose_4/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze?
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?
'sequential_5/conv1d_transpose_4/BiasAddBiasAddAsequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0>sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_5/conv1d_transpose_4/BiasAdd?
Bsequential_5/batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_5/batch_normalization_16/moments/mean/reduction_indices?
0sequential_5/batch_normalization_16/moments/meanMean0sequential_5/conv1d_transpose_4/BiasAdd:output:0Ksequential_5/batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_5/batch_normalization_16/moments/mean?
8sequential_5/batch_normalization_16/moments/StopGradientStopGradient9sequential_5/batch_normalization_16/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_5/batch_normalization_16/moments/StopGradient?
=sequential_5/batch_normalization_16/moments/SquaredDifferenceSquaredDifference0sequential_5/conv1d_transpose_4/BiasAdd:output:0Asequential_5/batch_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_5/batch_normalization_16/moments/SquaredDifference?
Fsequential_5/batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_5/batch_normalization_16/moments/variance/reduction_indices?
4sequential_5/batch_normalization_16/moments/varianceMeanAsequential_5/batch_normalization_16/moments/SquaredDifference:z:0Osequential_5/batch_normalization_16/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_5/batch_normalization_16/moments/variance?
3sequential_5/batch_normalization_16/moments/SqueezeSqueeze9sequential_5/batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_5/batch_normalization_16/moments/Squeeze?
5sequential_5/batch_normalization_16/moments/Squeeze_1Squeeze=sequential_5/batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_5/batch_normalization_16/moments/Squeeze_1?
9sequential_5/batch_normalization_16/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_16/AssignMovingAvg/15900*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_5/batch_normalization_16/AssignMovingAvg/decay?
Bsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_5_batch_normalization_16_assignmovingavg_15900*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOp?
7sequential_5/batch_normalization_16/AssignMovingAvg/subSubJsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0<sequential_5/batch_normalization_16/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_16/AssignMovingAvg/15900*
_output_shapes
:29
7sequential_5/batch_normalization_16/AssignMovingAvg/sub?
7sequential_5/batch_normalization_16/AssignMovingAvg/mulMul;sequential_5/batch_normalization_16/AssignMovingAvg/sub:z:0Bsequential_5/batch_normalization_16/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_16/AssignMovingAvg/15900*
_output_shapes
:29
7sequential_5/batch_normalization_16/AssignMovingAvg/mul?
Gsequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_5_batch_normalization_16_assignmovingavg_15900;sequential_5/batch_normalization_16/AssignMovingAvg/mul:z:0C^sequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_16/AssignMovingAvg/15900*
_output_shapes
 *
dtype02I
Gsequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?
;sequential_5/batch_normalization_16/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_16/AssignMovingAvg_1/15906*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_5/batch_normalization_16/AssignMovingAvg_1/decay?
Dsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_16_assignmovingavg_1_15906*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?
9sequential_5/batch_normalization_16/AssignMovingAvg_1/subSubLsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_5/batch_normalization_16/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_16/AssignMovingAvg_1/15906*
_output_shapes
:2;
9sequential_5/batch_normalization_16/AssignMovingAvg_1/sub?
9sequential_5/batch_normalization_16/AssignMovingAvg_1/mulMul=sequential_5/batch_normalization_16/AssignMovingAvg_1/sub:z:0Dsequential_5/batch_normalization_16/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_16/AssignMovingAvg_1/15906*
_output_shapes
:2;
9sequential_5/batch_normalization_16/AssignMovingAvg_1/mul?
Isequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_5_batch_normalization_16_assignmovingavg_1_15906=sequential_5/batch_normalization_16/AssignMovingAvg_1/mul:z:0E^sequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_16/AssignMovingAvg_1/15906*
_output_shapes
 *
dtype02K
Isequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_5/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_16/batchnorm/add/y?
1sequential_5/batch_normalization_16/batchnorm/addAddV2>sequential_5/batch_normalization_16/moments/Squeeze_1:output:0<sequential_5/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/add?
3sequential_5/batch_normalization_16/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_16/batchnorm/Rsqrt?
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_16/batchnorm/mulMul7sequential_5/batch_normalization_16/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/mul?
3sequential_5/batch_normalization_16/batchnorm/mul_1Mul0sequential_5/conv1d_transpose_4/BiasAdd:output:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_16/batchnorm/mul_1?
3sequential_5/batch_normalization_16/batchnorm/mul_2Mul<sequential_5/batch_normalization_16/moments/Squeeze:output:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_16/batchnorm/mul_2?
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?
1sequential_5/batch_normalization_16/batchnorm/subSubDsequential_5/batch_normalization_16/batchnorm/ReadVariableOp:value:07sequential_5/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/sub?
3sequential_5/batch_normalization_16/batchnorm/add_1AddV27sequential_5/batch_normalization_16/batchnorm/mul_1:z:05sequential_5/batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_16/batchnorm/add_1?
sequential_5/re_lu_7/ReluRelu7sequential_5/batch_normalization_16/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_5/re_lu_7/Relu?
%sequential_5/conv1d_transpose_5/ShapeShape'sequential_5/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_5/Shape?
3sequential_5/conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv1d_transpose_5/strided_slice/stack?
5sequential_5/conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice/stack_1?
5sequential_5/conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice/stack_2?
-sequential_5/conv1d_transpose_5/strided_sliceStridedSlice.sequential_5/conv1d_transpose_5/Shape:output:0<sequential_5/conv1d_transpose_5/strided_slice/stack:output:0>sequential_5/conv1d_transpose_5/strided_slice/stack_1:output:0>sequential_5/conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv1d_transpose_5/strided_slice?
5sequential_5/conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice_1/stack?
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_1?
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_2?
/sequential_5/conv1d_transpose_5/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_5/Shape:output:0>sequential_5/conv1d_transpose_5/strided_slice_1/stack:output:0@sequential_5/conv1d_transpose_5/strided_slice_1/stack_1:output:0@sequential_5/conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv1d_transpose_5/strided_slice_1?
%sequential_5/conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/conv1d_transpose_5/mul/y?
#sequential_5/conv1d_transpose_5/mulMul8sequential_5/conv1d_transpose_5/strided_slice_1:output:0.sequential_5/conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_5/conv1d_transpose_5/mul?
'sequential_5/conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv1d_transpose_5/stack/2?
%sequential_5/conv1d_transpose_5/stackPack6sequential_5/conv1d_transpose_5/strided_slice:output:0'sequential_5/conv1d_transpose_5/mul:z:00sequential_5/conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_5/stack?
?sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim?
;sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDims'sequential_5/re_lu_7/Relu:activations:0Hsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims?
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim?
=sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1?
Dsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2?
>sequential_5/conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice.sequential_5/conv1d_transpose_5/stack:output:0Msequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack?
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2?
@sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_5/stack:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1?
@sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1?
<sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis?
7sequential_5/conv1d_transpose_5/conv1d_transpose/concatConcatV2Gsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice:output:0Isequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0Isequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:0Esequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_5/conv1d_transpose_5/conv1d_transpose/concat?
0sequential_5/conv1d_transpose_5/conv1d_transposeConv2DBackpropInput@sequential_5/conv1d_transpose_5/conv1d_transpose/concat:output:0Fsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:0Dsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_5/conv1d_transpose_5/conv1d_transpose?
8sequential_5/conv1d_transpose_5/conv1d_transpose/SqueezeSqueeze9sequential_5/conv1d_transpose_5/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze?
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?
'sequential_5/conv1d_transpose_5/BiasAddBiasAddAsequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze:output:0>sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_5/conv1d_transpose_5/BiasAdd?
Bsequential_5/batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_5/batch_normalization_17/moments/mean/reduction_indices?
0sequential_5/batch_normalization_17/moments/meanMean0sequential_5/conv1d_transpose_5/BiasAdd:output:0Ksequential_5/batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_5/batch_normalization_17/moments/mean?
8sequential_5/batch_normalization_17/moments/StopGradientStopGradient9sequential_5/batch_normalization_17/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_5/batch_normalization_17/moments/StopGradient?
=sequential_5/batch_normalization_17/moments/SquaredDifferenceSquaredDifference0sequential_5/conv1d_transpose_5/BiasAdd:output:0Asequential_5/batch_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_5/batch_normalization_17/moments/SquaredDifference?
Fsequential_5/batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_5/batch_normalization_17/moments/variance/reduction_indices?
4sequential_5/batch_normalization_17/moments/varianceMeanAsequential_5/batch_normalization_17/moments/SquaredDifference:z:0Osequential_5/batch_normalization_17/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_5/batch_normalization_17/moments/variance?
3sequential_5/batch_normalization_17/moments/SqueezeSqueeze9sequential_5/batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_5/batch_normalization_17/moments/Squeeze?
5sequential_5/batch_normalization_17/moments/Squeeze_1Squeeze=sequential_5/batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_5/batch_normalization_17/moments/Squeeze_1?
9sequential_5/batch_normalization_17/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_17/AssignMovingAvg/15968*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_5/batch_normalization_17/AssignMovingAvg/decay?
Bsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_5_batch_normalization_17_assignmovingavg_15968*
_output_shapes
:*
dtype02D
Bsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOp?
7sequential_5/batch_normalization_17/AssignMovingAvg/subSubJsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0<sequential_5/batch_normalization_17/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_17/AssignMovingAvg/15968*
_output_shapes
:29
7sequential_5/batch_normalization_17/AssignMovingAvg/sub?
7sequential_5/batch_normalization_17/AssignMovingAvg/mulMul;sequential_5/batch_normalization_17/AssignMovingAvg/sub:z:0Bsequential_5/batch_normalization_17/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_5/batch_normalization_17/AssignMovingAvg/15968*
_output_shapes
:29
7sequential_5/batch_normalization_17/AssignMovingAvg/mul?
Gsequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_5_batch_normalization_17_assignmovingavg_15968;sequential_5/batch_normalization_17/AssignMovingAvg/mul:z:0C^sequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_5/batch_normalization_17/AssignMovingAvg/15968*
_output_shapes
 *
dtype02I
Gsequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?
;sequential_5/batch_normalization_17/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_17/AssignMovingAvg_1/15974*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_5/batch_normalization_17/AssignMovingAvg_1/decay?
Dsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_17_assignmovingavg_1_15974*
_output_shapes
:*
dtype02F
Dsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?
9sequential_5/batch_normalization_17/AssignMovingAvg_1/subSubLsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_5/batch_normalization_17/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_17/AssignMovingAvg_1/15974*
_output_shapes
:2;
9sequential_5/batch_normalization_17/AssignMovingAvg_1/sub?
9sequential_5/batch_normalization_17/AssignMovingAvg_1/mulMul=sequential_5/batch_normalization_17/AssignMovingAvg_1/sub:z:0Dsequential_5/batch_normalization_17/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_5/batch_normalization_17/AssignMovingAvg_1/15974*
_output_shapes
:2;
9sequential_5/batch_normalization_17/AssignMovingAvg_1/mul?
Isequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_5_batch_normalization_17_assignmovingavg_1_15974=sequential_5/batch_normalization_17/AssignMovingAvg_1/mul:z:0E^sequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_5/batch_normalization_17/AssignMovingAvg_1/15974*
_output_shapes
 *
dtype02K
Isequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_5/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_17/batchnorm/add/y?
1sequential_5/batch_normalization_17/batchnorm/addAddV2>sequential_5/batch_normalization_17/moments/Squeeze_1:output:0<sequential_5/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/add?
3sequential_5/batch_normalization_17/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_17/batchnorm/Rsqrt?
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_17/batchnorm/mulMul7sequential_5/batch_normalization_17/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/mul?
3sequential_5/batch_normalization_17/batchnorm/mul_1Mul0sequential_5/conv1d_transpose_5/BiasAdd:output:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_17/batchnorm/mul_1?
3sequential_5/batch_normalization_17/batchnorm/mul_2Mul<sequential_5/batch_normalization_17/moments/Squeeze:output:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_17/batchnorm/mul_2?
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?
1sequential_5/batch_normalization_17/batchnorm/subSubDsequential_5/batch_normalization_17/batchnorm/ReadVariableOp:value:07sequential_5/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/sub?
3sequential_5/batch_normalization_17/batchnorm/add_1AddV27sequential_5/batch_normalization_17/batchnorm/mul_1:z:05sequential_5/batch_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_17/batchnorm/add_1?
sequential_5/re_lu_8/ReluRelu7sequential_5/batch_normalization_17/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_5/re_lu_8/Relu?
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_5/flatten_5/Const?
sequential_5/flatten_5/ReshapeReshape'sequential_5/re_lu_8/Relu:activations:0%sequential_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_5/flatten_5/Reshape?
+sequential_5/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_5/dense_11/MatMul/ReadVariableOp?
sequential_5/dense_11/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_11/MatMul?
sequential_5/dense_11/TanhTanh&sequential_5/dense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_11/Tanh?
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_4/dense_8/MatMul/ReadVariableOp?
sequential_4/dense_8/MatMulMatMulsequential_5/dense_11/Tanh:y:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_8/MatMul?
Bsequential_4/batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_4/batch_normalization_12/moments/mean/reduction_indices?
0sequential_4/batch_normalization_12/moments/meanMean%sequential_4/dense_8/MatMul:product:0Ksequential_4/batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(22
0sequential_4/batch_normalization_12/moments/mean?
8sequential_4/batch_normalization_12/moments/StopGradientStopGradient9sequential_4/batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:2:
8sequential_4/batch_normalization_12/moments/StopGradient?
=sequential_4/batch_normalization_12/moments/SquaredDifferenceSquaredDifference%sequential_4/dense_8/MatMul:product:0Asequential_4/batch_normalization_12/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2?
=sequential_4/batch_normalization_12/moments/SquaredDifference?
Fsequential_4/batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_4/batch_normalization_12/moments/variance/reduction_indices?
4sequential_4/batch_normalization_12/moments/varianceMeanAsequential_4/batch_normalization_12/moments/SquaredDifference:z:0Osequential_4/batch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(26
4sequential_4/batch_normalization_12/moments/variance?
3sequential_4/batch_normalization_12/moments/SqueezeSqueeze9sequential_4/batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_4/batch_normalization_12/moments/Squeeze?
5sequential_4/batch_normalization_12/moments/Squeeze_1Squeeze=sequential_4/batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_4/batch_normalization_12/moments/Squeeze_1?
9sequential_4/batch_normalization_12/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_12/AssignMovingAvg/16010*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_4/batch_normalization_12/AssignMovingAvg/decay?
Bsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_4_batch_normalization_12_assignmovingavg_16010*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOp?
7sequential_4/batch_normalization_12/AssignMovingAvg/subSubJsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0<sequential_4/batch_normalization_12/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_12/AssignMovingAvg/16010*
_output_shapes
:29
7sequential_4/batch_normalization_12/AssignMovingAvg/sub?
7sequential_4/batch_normalization_12/AssignMovingAvg/mulMul;sequential_4/batch_normalization_12/AssignMovingAvg/sub:z:0Bsequential_4/batch_normalization_12/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_12/AssignMovingAvg/16010*
_output_shapes
:29
7sequential_4/batch_normalization_12/AssignMovingAvg/mul?
Gsequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_4_batch_normalization_12_assignmovingavg_16010;sequential_4/batch_normalization_12/AssignMovingAvg/mul:z:0C^sequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_12/AssignMovingAvg/16010*
_output_shapes
 *
dtype02I
Gsequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?
;sequential_4/batch_normalization_12/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_12/AssignMovingAvg_1/16016*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_4/batch_normalization_12/AssignMovingAvg_1/decay?
Dsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_12_assignmovingavg_1_16016*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?
9sequential_4/batch_normalization_12/AssignMovingAvg_1/subSubLsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_4/batch_normalization_12/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_12/AssignMovingAvg_1/16016*
_output_shapes
:2;
9sequential_4/batch_normalization_12/AssignMovingAvg_1/sub?
9sequential_4/batch_normalization_12/AssignMovingAvg_1/mulMul=sequential_4/batch_normalization_12/AssignMovingAvg_1/sub:z:0Dsequential_4/batch_normalization_12/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_12/AssignMovingAvg_1/16016*
_output_shapes
:2;
9sequential_4/batch_normalization_12/AssignMovingAvg_1/mul?
Isequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_4_batch_normalization_12_assignmovingavg_1_16016=sequential_4/batch_normalization_12/AssignMovingAvg_1/mul:z:0E^sequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_12/AssignMovingAvg_1/16016*
_output_shapes
 *
dtype02K
Isequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_4/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_12/batchnorm/add/y?
1sequential_4/batch_normalization_12/batchnorm/addAddV2>sequential_4/batch_normalization_12/moments/Squeeze_1:output:0<sequential_4/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/add?
3sequential_4/batch_normalization_12/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_12/batchnorm/Rsqrt?
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_12/batchnorm/mulMul7sequential_4/batch_normalization_12/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/mul?
3sequential_4/batch_normalization_12/batchnorm/mul_1Mul%sequential_4/dense_8/MatMul:product:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????25
3sequential_4/batch_normalization_12/batchnorm/mul_1?
3sequential_4/batch_normalization_12/batchnorm/mul_2Mul<sequential_4/batch_normalization_12/moments/Squeeze:output:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_12/batchnorm/mul_2?
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?
1sequential_4/batch_normalization_12/batchnorm/subSubDsequential_4/batch_normalization_12/batchnorm/ReadVariableOp:value:07sequential_4/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/sub?
3sequential_4/batch_normalization_12/batchnorm/add_1AddV27sequential_4/batch_normalization_12/batchnorm/mul_1:z:05sequential_4/batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????25
3sequential_4/batch_normalization_12/batchnorm/add_1?
$sequential_4/leaky_re_lu_6/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_12/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_6/LeakyRelu?
sequential_4/reshape_4/ShapeShape2sequential_4/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_4/reshape_4/Shape?
*sequential_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_4/strided_slice/stack?
,sequential_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_4/strided_slice/stack_1?
,sequential_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_4/strided_slice/stack_2?
$sequential_4/reshape_4/strided_sliceStridedSlice%sequential_4/reshape_4/Shape:output:03sequential_4/reshape_4/strided_slice/stack:output:05sequential_4/reshape_4/strided_slice/stack_1:output:05sequential_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_4/strided_slice?
&sequential_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_4/Reshape/shape/1?
&sequential_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_4/Reshape/shape/2?
$sequential_4/reshape_4/Reshape/shapePack-sequential_4/reshape_4/strided_slice:output:0/sequential_4/reshape_4/Reshape/shape/1:output:0/sequential_4/reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_4/Reshape/shape?
sequential_4/reshape_4/ReshapeReshape2sequential_4/leaky_re_lu_6/LeakyRelu:activations:0-sequential_4/reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_4/reshape_4/Reshape?
+sequential_4/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_4/conv1d_4/conv1d/ExpandDims/dim?
'sequential_4/conv1d_4/conv1d/ExpandDims
ExpandDims'sequential_4/reshape_4/Reshape:output:04sequential_4/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_4/conv1d_4/conv1d/ExpandDims?
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_4/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_4/conv1d_4/conv1d/ExpandDims_1/dim?
)sequential_4/conv1d_4/conv1d/ExpandDims_1
ExpandDims@sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_4/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_4/conv1d_4/conv1d/ExpandDims_1?
sequential_4/conv1d_4/conv1dConv2D0sequential_4/conv1d_4/conv1d/ExpandDims:output:02sequential_4/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_4/conv1d_4/conv1d?
$sequential_4/conv1d_4/conv1d/SqueezeSqueeze%sequential_4/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_4/conv1d_4/conv1d/Squeeze?
,sequential_4/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/conv1d_4/BiasAdd/ReadVariableOp?
sequential_4/conv1d_4/BiasAddBiasAdd-sequential_4/conv1d_4/conv1d/Squeeze:output:04sequential_4/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_4/conv1d_4/BiasAdd?
Bsequential_4/batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_4/batch_normalization_13/moments/mean/reduction_indices?
0sequential_4/batch_normalization_13/moments/meanMean&sequential_4/conv1d_4/BiasAdd:output:0Ksequential_4/batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_4/batch_normalization_13/moments/mean?
8sequential_4/batch_normalization_13/moments/StopGradientStopGradient9sequential_4/batch_normalization_13/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_4/batch_normalization_13/moments/StopGradient?
=sequential_4/batch_normalization_13/moments/SquaredDifferenceSquaredDifference&sequential_4/conv1d_4/BiasAdd:output:0Asequential_4/batch_normalization_13/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_4/batch_normalization_13/moments/SquaredDifference?
Fsequential_4/batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_4/batch_normalization_13/moments/variance/reduction_indices?
4sequential_4/batch_normalization_13/moments/varianceMeanAsequential_4/batch_normalization_13/moments/SquaredDifference:z:0Osequential_4/batch_normalization_13/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_4/batch_normalization_13/moments/variance?
3sequential_4/batch_normalization_13/moments/SqueezeSqueeze9sequential_4/batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_4/batch_normalization_13/moments/Squeeze?
5sequential_4/batch_normalization_13/moments/Squeeze_1Squeeze=sequential_4/batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_4/batch_normalization_13/moments/Squeeze_1?
9sequential_4/batch_normalization_13/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_13/AssignMovingAvg/16063*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_4/batch_normalization_13/AssignMovingAvg/decay?
Bsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_4_batch_normalization_13_assignmovingavg_16063*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOp?
7sequential_4/batch_normalization_13/AssignMovingAvg/subSubJsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0<sequential_4/batch_normalization_13/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_13/AssignMovingAvg/16063*
_output_shapes
:29
7sequential_4/batch_normalization_13/AssignMovingAvg/sub?
7sequential_4/batch_normalization_13/AssignMovingAvg/mulMul;sequential_4/batch_normalization_13/AssignMovingAvg/sub:z:0Bsequential_4/batch_normalization_13/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_13/AssignMovingAvg/16063*
_output_shapes
:29
7sequential_4/batch_normalization_13/AssignMovingAvg/mul?
Gsequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_4_batch_normalization_13_assignmovingavg_16063;sequential_4/batch_normalization_13/AssignMovingAvg/mul:z:0C^sequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_13/AssignMovingAvg/16063*
_output_shapes
 *
dtype02I
Gsequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOp?
;sequential_4/batch_normalization_13/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_13/AssignMovingAvg_1/16069*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_4/batch_normalization_13/AssignMovingAvg_1/decay?
Dsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_13_assignmovingavg_1_16069*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp?
9sequential_4/batch_normalization_13/AssignMovingAvg_1/subSubLsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_4/batch_normalization_13/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_13/AssignMovingAvg_1/16069*
_output_shapes
:2;
9sequential_4/batch_normalization_13/AssignMovingAvg_1/sub?
9sequential_4/batch_normalization_13/AssignMovingAvg_1/mulMul=sequential_4/batch_normalization_13/AssignMovingAvg_1/sub:z:0Dsequential_4/batch_normalization_13/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_13/AssignMovingAvg_1/16069*
_output_shapes
:2;
9sequential_4/batch_normalization_13/AssignMovingAvg_1/mul?
Isequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_4_batch_normalization_13_assignmovingavg_1_16069=sequential_4/batch_normalization_13/AssignMovingAvg_1/mul:z:0E^sequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_13/AssignMovingAvg_1/16069*
_output_shapes
 *
dtype02K
Isequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_4/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_13/batchnorm/add/y?
1sequential_4/batch_normalization_13/batchnorm/addAddV2>sequential_4/batch_normalization_13/moments/Squeeze_1:output:0<sequential_4/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/add?
3sequential_4/batch_normalization_13/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_13/batchnorm/Rsqrt?
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_13/batchnorm/mulMul7sequential_4/batch_normalization_13/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/mul?
3sequential_4/batch_normalization_13/batchnorm/mul_1Mul&sequential_4/conv1d_4/BiasAdd:output:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_13/batchnorm/mul_1?
3sequential_4/batch_normalization_13/batchnorm/mul_2Mul<sequential_4/batch_normalization_13/moments/Squeeze:output:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_13/batchnorm/mul_2?
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?
1sequential_4/batch_normalization_13/batchnorm/subSubDsequential_4/batch_normalization_13/batchnorm/ReadVariableOp:value:07sequential_4/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/sub?
3sequential_4/batch_normalization_13/batchnorm/add_1AddV27sequential_4/batch_normalization_13/batchnorm/mul_1:z:05sequential_4/batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_13/batchnorm/add_1?
$sequential_4/leaky_re_lu_7/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_13/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_7/LeakyRelu?
+sequential_4/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_4/conv1d_5/conv1d/ExpandDims/dim?
'sequential_4/conv1d_5/conv1d/ExpandDims
ExpandDims2sequential_4/leaky_re_lu_7/LeakyRelu:activations:04sequential_4/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_4/conv1d_5/conv1d/ExpandDims?
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_4/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_4/conv1d_5/conv1d/ExpandDims_1/dim?
)sequential_4/conv1d_5/conv1d/ExpandDims_1
ExpandDims@sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_4/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_4/conv1d_5/conv1d/ExpandDims_1?
sequential_4/conv1d_5/conv1dConv2D0sequential_4/conv1d_5/conv1d/ExpandDims:output:02sequential_4/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_4/conv1d_5/conv1d?
$sequential_4/conv1d_5/conv1d/SqueezeSqueeze%sequential_4/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_4/conv1d_5/conv1d/Squeeze?
,sequential_4/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/conv1d_5/BiasAdd/ReadVariableOp?
sequential_4/conv1d_5/BiasAddBiasAdd-sequential_4/conv1d_5/conv1d/Squeeze:output:04sequential_4/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_4/conv1d_5/BiasAdd?
Bsequential_4/batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_4/batch_normalization_14/moments/mean/reduction_indices?
0sequential_4/batch_normalization_14/moments/meanMean&sequential_4/conv1d_5/BiasAdd:output:0Ksequential_4/batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_4/batch_normalization_14/moments/mean?
8sequential_4/batch_normalization_14/moments/StopGradientStopGradient9sequential_4/batch_normalization_14/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_4/batch_normalization_14/moments/StopGradient?
=sequential_4/batch_normalization_14/moments/SquaredDifferenceSquaredDifference&sequential_4/conv1d_5/BiasAdd:output:0Asequential_4/batch_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_4/batch_normalization_14/moments/SquaredDifference?
Fsequential_4/batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_4/batch_normalization_14/moments/variance/reduction_indices?
4sequential_4/batch_normalization_14/moments/varianceMeanAsequential_4/batch_normalization_14/moments/SquaredDifference:z:0Osequential_4/batch_normalization_14/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_4/batch_normalization_14/moments/variance?
3sequential_4/batch_normalization_14/moments/SqueezeSqueeze9sequential_4/batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_4/batch_normalization_14/moments/Squeeze?
5sequential_4/batch_normalization_14/moments/Squeeze_1Squeeze=sequential_4/batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_4/batch_normalization_14/moments/Squeeze_1?
9sequential_4/batch_normalization_14/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_14/AssignMovingAvg/16107*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_4/batch_normalization_14/AssignMovingAvg/decay?
Bsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp9sequential_4_batch_normalization_14_assignmovingavg_16107*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOp?
7sequential_4/batch_normalization_14/AssignMovingAvg/subSubJsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0<sequential_4/batch_normalization_14/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_14/AssignMovingAvg/16107*
_output_shapes
:29
7sequential_4/batch_normalization_14/AssignMovingAvg/sub?
7sequential_4/batch_normalization_14/AssignMovingAvg/mulMul;sequential_4/batch_normalization_14/AssignMovingAvg/sub:z:0Bsequential_4/batch_normalization_14/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_4/batch_normalization_14/AssignMovingAvg/16107*
_output_shapes
:29
7sequential_4/batch_normalization_14/AssignMovingAvg/mul?
Gsequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp9sequential_4_batch_normalization_14_assignmovingavg_16107;sequential_4/batch_normalization_14/AssignMovingAvg/mul:z:0C^sequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_4/batch_normalization_14/AssignMovingAvg/16107*
_output_shapes
 *
dtype02I
Gsequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOp?
;sequential_4/batch_normalization_14/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_14/AssignMovingAvg_1/16113*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_4/batch_normalization_14/AssignMovingAvg_1/decay?
Dsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_14_assignmovingavg_1_16113*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp?
9sequential_4/batch_normalization_14/AssignMovingAvg_1/subSubLsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_4/batch_normalization_14/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_14/AssignMovingAvg_1/16113*
_output_shapes
:2;
9sequential_4/batch_normalization_14/AssignMovingAvg_1/sub?
9sequential_4/batch_normalization_14/AssignMovingAvg_1/mulMul=sequential_4/batch_normalization_14/AssignMovingAvg_1/sub:z:0Dsequential_4/batch_normalization_14/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@sequential_4/batch_normalization_14/AssignMovingAvg_1/16113*
_output_shapes
:2;
9sequential_4/batch_normalization_14/AssignMovingAvg_1/mul?
Isequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp;sequential_4_batch_normalization_14_assignmovingavg_1_16113=sequential_4/batch_normalization_14/AssignMovingAvg_1/mul:z:0E^sequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@sequential_4/batch_normalization_14/AssignMovingAvg_1/16113*
_output_shapes
 *
dtype02K
Isequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_4/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_14/batchnorm/add/y?
1sequential_4/batch_normalization_14/batchnorm/addAddV2>sequential_4/batch_normalization_14/moments/Squeeze_1:output:0<sequential_4/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/add?
3sequential_4/batch_normalization_14/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_14/batchnorm/Rsqrt?
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_14/batchnorm/mulMul7sequential_4/batch_normalization_14/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/mul?
3sequential_4/batch_normalization_14/batchnorm/mul_1Mul&sequential_4/conv1d_5/BiasAdd:output:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_14/batchnorm/mul_1?
3sequential_4/batch_normalization_14/batchnorm/mul_2Mul<sequential_4/batch_normalization_14/moments/Squeeze:output:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_14/batchnorm/mul_2?
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?
1sequential_4/batch_normalization_14/batchnorm/subSubDsequential_4/batch_normalization_14/batchnorm/ReadVariableOp:value:07sequential_4/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/sub?
3sequential_4/batch_normalization_14/batchnorm/add_1AddV27sequential_4/batch_normalization_14/batchnorm/mul_1:z:05sequential_4/batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_14/batchnorm/add_1?
$sequential_4/leaky_re_lu_8/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_14/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_8/LeakyRelu?
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_4/flatten_4/Const?
sequential_4/flatten_4/ReshapeReshape2sequential_4/leaky_re_lu_8/LeakyRelu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_4/flatten_4/Reshape?
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_4/dense_9/MatMul/ReadVariableOp?
sequential_4/dense_9/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_9/MatMul?
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_9/BiasAdd/ReadVariableOp?
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_9/BiasAdd?
IdentityIdentity%sequential_4/dense_9/BiasAdd:output:0H^sequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOpC^sequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOpJ^sequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpE^sequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp=^sequential_4/batch_normalization_12/batchnorm/ReadVariableOpA^sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpH^sequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOpC^sequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOpJ^sequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpE^sequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp=^sequential_4/batch_normalization_13/batchnorm/ReadVariableOpA^sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpH^sequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOpC^sequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOpJ^sequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpE^sequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp=^sequential_4/batch_normalization_14/batchnorm/ReadVariableOpA^sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp-^sequential_4/conv1d_4/BiasAdd/ReadVariableOp9^sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp-^sequential_4/conv1d_5/BiasAdd/ReadVariableOp9^sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOp,^sequential_4/dense_9/BiasAdd/ReadVariableOp+^sequential_4/dense_9/MatMul/ReadVariableOpH^sequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOpC^sequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOpJ^sequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpE^sequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp=^sequential_5/batch_normalization_15/batchnorm/ReadVariableOpA^sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpH^sequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOpC^sequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOpJ^sequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpE^sequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp=^sequential_5/batch_normalization_16/batchnorm/ReadVariableOpA^sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpH^sequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOpC^sequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOpJ^sequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpE^sequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp=^sequential_5/batch_normalization_17/batchnorm/ReadVariableOpA^sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp7^sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpM^sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp7^sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpM^sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp,^sequential_5/dense_10/MatMul/ReadVariableOp,^sequential_5/dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2?
Gsequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOpGsequential_4/batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOpBsequential_4/batch_normalization_12/AssignMovingAvg/ReadVariableOp2?
Isequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpIsequential_4/batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOpDsequential_4/batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2|
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp2?
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp2?
Gsequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOpGsequential_4/batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOpBsequential_4/batch_normalization_13/AssignMovingAvg/ReadVariableOp2?
Isequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpIsequential_4/batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOpDsequential_4/batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2|
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp2?
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp2?
Gsequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOpGsequential_4/batch_normalization_14/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOpBsequential_4/batch_normalization_14/AssignMovingAvg/ReadVariableOp2?
Isequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpIsequential_4/batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOpDsequential_4/batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2|
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp2?
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp2\
,sequential_4/conv1d_4/BiasAdd/ReadVariableOp,sequential_4/conv1d_4/BiasAdd/ReadVariableOp2t
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2\
,sequential_4/conv1d_5/BiasAdd/ReadVariableOp,sequential_4/conv1d_5/BiasAdd/ReadVariableOp2t
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2Z
+sequential_4/dense_9/BiasAdd/ReadVariableOp+sequential_4/dense_9/BiasAdd/ReadVariableOp2X
*sequential_4/dense_9/MatMul/ReadVariableOp*sequential_4/dense_9/MatMul/ReadVariableOp2?
Gsequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOpGsequential_5/batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOpBsequential_5/batch_normalization_15/AssignMovingAvg/ReadVariableOp2?
Isequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpIsequential_5/batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOpDsequential_5/batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2|
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp2?
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp2?
Gsequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOpGsequential_5/batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOpBsequential_5/batch_normalization_16/AssignMovingAvg/ReadVariableOp2?
Isequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpIsequential_5/batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOpDsequential_5/batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2|
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp2?
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp2?
Gsequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOpGsequential_5/batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOpBsequential_5/batch_normalization_17/AssignMovingAvg/ReadVariableOp2?
Isequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpIsequential_5/batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOpDsequential_5/batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2|
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp2?
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp2p
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp2?
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2p
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp2?
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2Z
+sequential_5/dense_10/MatMul/ReadVariableOp+sequential_5/dense_10/MatMul/ReadVariableOp2Z
+sequential_5/dense_11/MatMul/ReadVariableOp+sequential_5/dense_11/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18083

inputs
assignmovingavg_18058
assignmovingavg_1_18064)
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
T0*+
_output_shapes
:?????????2
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/18058*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_18058*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/18058*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/18058*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_18058AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/18058*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/18064*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_18064*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/18064*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/18064*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_18064AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/18064*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_12_layer_call_fn_17715

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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_141372
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
6__inference_batch_normalization_13_layer_call_fn_17918

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_142442
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
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_17599

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
?
?
,__inference_sequential_4_layer_call_fn_17279

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

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_150392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17551

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
?
?
6__inference_batch_normalization_14_layer_call_fn_18129

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
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_147412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_17738

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
?
?
C__inference_dense_10_layer_call_and_return_conditional_losses_17286

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
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14277

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
??
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_17193

inputs*
&dense_8_matmul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource@
<batch_normalization_12_batchnorm_mul_readvariableop_resource>
:batch_normalization_12_batchnorm_readvariableop_1_resource>
:batch_normalization_12_batchnorm_readvariableop_2_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource@
<batch_normalization_13_batchnorm_mul_readvariableop_resource>
:batch_normalization_13_batchnorm_readvariableop_1_resource>
:batch_normalization_13_batchnorm_readvariableop_2_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource<
8batch_normalization_14_batchnorm_readvariableop_resource@
<batch_normalization_14_batchnorm_mul_readvariableop_resource>
:batch_normalization_14_batchnorm_readvariableop_1_resource>
:batch_normalization_14_batchnorm_readvariableop_2_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??/batch_normalization_12/batchnorm/ReadVariableOp?1batch_normalization_12/batchnorm/ReadVariableOp_1?1batch_normalization_12/batchnorm/ReadVariableOp_2?3batch_normalization_12/batchnorm/mul/ReadVariableOp?/batch_normalization_13/batchnorm/ReadVariableOp?1batch_normalization_13/batchnorm/ReadVariableOp_1?1batch_normalization_13/batchnorm/ReadVariableOp_2?3batch_normalization_13/batchnorm/mul/ReadVariableOp?/batch_normalization_14/batchnorm/ReadVariableOp?1batch_normalization_14/batchnorm/ReadVariableOp_1?1batch_normalization_14/batchnorm/ReadVariableOp_2?3batch_normalization_14/batchnorm/mul/ReadVariableOp?conv1d_4/BiasAdd/ReadVariableOp?+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?conv1d_5/BiasAdd/ReadVariableOp?+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/mul_1?
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1?
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2?
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2?
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/add_1?
leaky_re_lu_6/LeakyRelu	LeakyRelu*batch_normalization_12/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_6/LeakyReluw
reshape_4/ShapeShape%leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshape%leaky_re_lu_6/LeakyRelu:activations:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_4/Reshape?
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_4/conv1d/ExpandDims/dim?
conv1d_4/conv1d/ExpandDims
ExpandDimsreshape_4/Reshape:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_4/conv1d/ExpandDims?
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim?
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_4/conv1d/ExpandDims_1?
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_4/conv1d?
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_4/conv1d/Squeeze?
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp?
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_4/BiasAdd?
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOp?
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_13/batchnorm/add/y?
$batch_normalization_13/batchnorm/addAddV27batch_normalization_13/batchnorm/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add?
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt?
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOp?
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul?
&batch_normalization_13/batchnorm/mul_1Mulconv1d_4/BiasAdd:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_13/batchnorm/mul_1?
1batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_1?
&batch_normalization_13/batchnorm/mul_2Mul9batch_normalization_13/batchnorm/ReadVariableOp_1:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2?
1batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_2?
$batch_normalization_13/batchnorm/subSub9batch_normalization_13/batchnorm/ReadVariableOp_2:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub?
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_13/batchnorm/add_1?
leaky_re_lu_7/LeakyRelu	LeakyRelu*batch_normalization_13/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_7/LeakyRelu?
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_5/conv1d/ExpandDims/dim?
conv1d_5/conv1d/ExpandDims
ExpandDims%leaky_re_lu_7/LeakyRelu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_5/conv1d/ExpandDims?
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim?
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_5/conv1d/ExpandDims_1?
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_5/conv1d?
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_5/conv1d/Squeeze?
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp?
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_5/BiasAdd?
/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_14/batchnorm/ReadVariableOp?
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_14/batchnorm/add/y?
$batch_normalization_14/batchnorm/addAddV27batch_normalization_14/batchnorm/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add?
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt?
3batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_14/batchnorm/mul/ReadVariableOp?
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:0;batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul?
&batch_normalization_14/batchnorm/mul_1Mulconv1d_5/BiasAdd:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/mul_1?
1batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_14/batchnorm/ReadVariableOp_1?
&batch_normalization_14/batchnorm/mul_2Mul9batch_normalization_14/batchnorm/ReadVariableOp_1:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2?
1batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_14/batchnorm/ReadVariableOp_2?
$batch_normalization_14/batchnorm/subSub9batch_normalization_14/batchnorm/ReadVariableOp_2:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub?
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_14/batchnorm/add_1?
leaky_re_lu_8/LeakyRelu	LeakyRelu*batch_normalization_14/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_8/LeakyRelus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshape%leaky_re_lu_8/LeakyRelu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd?
IdentityIdentitydense_9/BiasAdd:output:00^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp2^batch_normalization_13/batchnorm/ReadVariableOp_12^batch_normalization_13/batchnorm/ReadVariableOp_24^batch_normalization_13/batchnorm/mul/ReadVariableOp0^batch_normalization_14/batchnorm/ReadVariableOp2^batch_normalization_14/batchnorm/ReadVariableOp_12^batch_normalization_14/batchnorm/ReadVariableOp_24^batch_normalization_14/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2f
1batch_normalization_13/batchnorm/ReadVariableOp_11batch_normalization_13/batchnorm/ReadVariableOp_12f
1batch_normalization_13/batchnorm/ReadVariableOp_21batch_normalization_13/batchnorm/ReadVariableOp_22j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2b
/batch_normalization_14/batchnorm/ReadVariableOp/batch_normalization_14/batchnorm/ReadVariableOp2f
1batch_normalization_14/batchnorm/ReadVariableOp_11batch_normalization_14/batchnorm/ReadVariableOp_12f
1batch_normalization_14/batchnorm/ReadVariableOp_21batch_normalization_14/batchnorm/ReadVariableOp_22j
3batch_normalization_14/batchnorm/mul/ReadVariableOp3batch_normalization_14/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_18424
file_prefix$
 assignvariableop_dense_10_kernel3
/assignvariableop_1_batch_normalization_15_gamma2
.assignvariableop_2_batch_normalization_15_beta0
,assignvariableop_3_conv1d_transpose_4_kernel.
*assignvariableop_4_conv1d_transpose_4_bias3
/assignvariableop_5_batch_normalization_16_gamma2
.assignvariableop_6_batch_normalization_16_beta0
,assignvariableop_7_conv1d_transpose_5_kernel.
*assignvariableop_8_conv1d_transpose_5_bias3
/assignvariableop_9_batch_normalization_17_gamma3
/assignvariableop_10_batch_normalization_17_beta'
#assignvariableop_11_dense_11_kernel&
"assignvariableop_12_dense_8_kernel4
0assignvariableop_13_batch_normalization_12_gamma3
/assignvariableop_14_batch_normalization_12_beta'
#assignvariableop_15_conv1d_4_kernel%
!assignvariableop_16_conv1d_4_bias4
0assignvariableop_17_batch_normalization_13_gamma3
/assignvariableop_18_batch_normalization_13_beta'
#assignvariableop_19_conv1d_5_kernel%
!assignvariableop_20_conv1d_5_bias4
0assignvariableop_21_batch_normalization_14_gamma3
/assignvariableop_22_batch_normalization_14_beta&
"assignvariableop_23_dense_9_kernel$
 assignvariableop_24_dense_9_bias:
6assignvariableop_25_batch_normalization_15_moving_mean>
:assignvariableop_26_batch_normalization_15_moving_variance:
6assignvariableop_27_batch_normalization_16_moving_mean>
:assignvariableop_28_batch_normalization_16_moving_variance:
6assignvariableop_29_batch_normalization_17_moving_mean>
:assignvariableop_30_batch_normalization_17_moving_variance:
6assignvariableop_31_batch_normalization_12_moving_mean>
:assignvariableop_32_batch_normalization_12_moving_variance:
6assignvariableop_33_batch_normalization_13_moving_mean>
:assignvariableop_34_batch_normalization_13_moving_variance:
6assignvariableop_35_batch_normalization_14_moving_mean>
:assignvariableop_36_batch_normalization_14_moving_variance
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_15_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_15_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_conv1d_transpose_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv1d_transpose_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp/assignvariableop_5_batch_normalization_16_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_16_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv1d_transpose_5_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv1d_transpose_5_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_17_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_17_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_11_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_8_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_12_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_12_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv1d_4_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv1d_4_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_13_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_13_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv1d_5_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv1d_5_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_14_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_14_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_9_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_9_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_batch_normalization_15_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp:assignvariableop_26_batch_normalization_15_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_16_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_16_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_batch_normalization_17_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp:assignvariableop_30_batch_normalization_17_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp6assignvariableop_31_batch_normalization_12_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp:assignvariableop_32_batch_normalization_12_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_batch_normalization_13_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp:assignvariableop_34_batch_normalization_13_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_batch_normalization_14_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp:assignvariableop_36_batch_normalization_14_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_36AssignVariableOp_362(
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
?
}
(__inference_conv1d_4_layer_call_fn_17767

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
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_145352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17905

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

?
,__inference_sequential_5_layer_call_fn_16910

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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_138762
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
?
?
6__inference_batch_normalization_15_layer_call_fn_17362

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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_131062
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
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_14741

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_12_layer_call_fn_17702

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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_141042
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
?
,__inference_sequential_4_layer_call_fn_15080
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

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_150392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_8_input
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17689

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
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14244

inputs
assignmovingavg_14219
assignmovingavg_1_14225)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14219*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14219*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14219*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14219*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14219AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14219*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14225*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14225*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14225*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14225*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14225AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14225*
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
?
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_14647

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???>2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?'
G__inference_sequential_6_layer_call_and_return_conditional_losses_16377

inputs8
4sequential_5_dense_10_matmul_readvariableop_resourceI
Esequential_5_batch_normalization_15_batchnorm_readvariableop_resourceM
Isequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resourceK
Gsequential_5_batch_normalization_15_batchnorm_readvariableop_1_resourceK
Gsequential_5_batch_normalization_15_batchnorm_readvariableop_2_resourceY
Usequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_5_conv1d_transpose_4_biasadd_readvariableop_resourceI
Esequential_5_batch_normalization_16_batchnorm_readvariableop_resourceM
Isequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resourceK
Gsequential_5_batch_normalization_16_batchnorm_readvariableop_1_resourceK
Gsequential_5_batch_normalization_16_batchnorm_readvariableop_2_resourceY
Usequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_5_conv1d_transpose_5_biasadd_readvariableop_resourceI
Esequential_5_batch_normalization_17_batchnorm_readvariableop_resourceM
Isequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resourceK
Gsequential_5_batch_normalization_17_batchnorm_readvariableop_1_resourceK
Gsequential_5_batch_normalization_17_batchnorm_readvariableop_2_resource8
4sequential_5_dense_11_matmul_readvariableop_resource7
3sequential_4_dense_8_matmul_readvariableop_resourceI
Esequential_4_batch_normalization_12_batchnorm_readvariableop_resourceM
Isequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resourceK
Gsequential_4_batch_normalization_12_batchnorm_readvariableop_1_resourceK
Gsequential_4_batch_normalization_12_batchnorm_readvariableop_2_resourceE
Asequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resource9
5sequential_4_conv1d_4_biasadd_readvariableop_resourceI
Esequential_4_batch_normalization_13_batchnorm_readvariableop_resourceM
Isequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resourceK
Gsequential_4_batch_normalization_13_batchnorm_readvariableop_1_resourceK
Gsequential_4_batch_normalization_13_batchnorm_readvariableop_2_resourceE
Asequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resource9
5sequential_4_conv1d_5_biasadd_readvariableop_resourceI
Esequential_4_batch_normalization_14_batchnorm_readvariableop_resourceM
Isequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resourceK
Gsequential_4_batch_normalization_14_batchnorm_readvariableop_1_resourceK
Gsequential_4_batch_normalization_14_batchnorm_readvariableop_2_resource7
3sequential_4_dense_9_matmul_readvariableop_resource8
4sequential_4_dense_9_biasadd_readvariableop_resource
identity??<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2?@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2?@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2?@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?,sequential_4/conv1d_4/BiasAdd/ReadVariableOp?8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?,sequential_4/conv1d_5/BiasAdd/ReadVariableOp?8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?*sequential_4/dense_8/MatMul/ReadVariableOp?+sequential_4/dense_9/BiasAdd/ReadVariableOp?*sequential_4/dense_9/MatMul/ReadVariableOp?<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1?>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2?@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1?>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2?@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1?>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2?@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?+sequential_5/dense_10/MatMul/ReadVariableOp?+sequential_5/dense_11/MatMul/ReadVariableOp?
+sequential_5/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_5/dense_10/MatMul/ReadVariableOp?
sequential_5/dense_10/MatMulMatMulinputs3sequential_5/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_10/MatMul?
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?
3sequential_5/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_15/batchnorm/add/y?
1sequential_5/batch_normalization_15/batchnorm/addAddV2Dsequential_5/batch_normalization_15/batchnorm/ReadVariableOp:value:0<sequential_5/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/add?
3sequential_5/batch_normalization_15/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_15/batchnorm/Rsqrt?
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_15/batchnorm/mulMul7sequential_5/batch_normalization_15/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/mul?
3sequential_5/batch_normalization_15/batchnorm/mul_1Mul&sequential_5/dense_10/MatMul:product:05sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????25
3sequential_5/batch_normalization_15/batchnorm/mul_1?
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_5_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1?
3sequential_5/batch_normalization_15/batchnorm/mul_2MulFsequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1:value:05sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_15/batchnorm/mul_2?
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_5_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2?
1sequential_5/batch_normalization_15/batchnorm/subSubFsequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2:value:07sequential_5/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_15/batchnorm/sub?
3sequential_5/batch_normalization_15/batchnorm/add_1AddV27sequential_5/batch_normalization_15/batchnorm/mul_1:z:05sequential_5/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????25
3sequential_5/batch_normalization_15/batchnorm/add_1?
sequential_5/re_lu_6/ReluRelu7sequential_5/batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
sequential_5/re_lu_6/Relu?
sequential_5/reshape_5/ShapeShape'sequential_5/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
sequential_5/reshape_5/Shape?
*sequential_5/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/reshape_5/strided_slice/stack?
,sequential_5/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_5/strided_slice/stack_1?
,sequential_5/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_5/strided_slice/stack_2?
$sequential_5/reshape_5/strided_sliceStridedSlice%sequential_5/reshape_5/Shape:output:03sequential_5/reshape_5/strided_slice/stack:output:05sequential_5/reshape_5/strided_slice/stack_1:output:05sequential_5/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/reshape_5/strided_slice?
&sequential_5/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_5/Reshape/shape/1?
&sequential_5/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_5/Reshape/shape/2?
$sequential_5/reshape_5/Reshape/shapePack-sequential_5/reshape_5/strided_slice:output:0/sequential_5/reshape_5/Reshape/shape/1:output:0/sequential_5/reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_5/reshape_5/Reshape/shape?
sequential_5/reshape_5/ReshapeReshape'sequential_5/re_lu_6/Relu:activations:0-sequential_5/reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_5/reshape_5/Reshape?
%sequential_5/conv1d_transpose_4/ShapeShape'sequential_5/reshape_5/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_4/Shape?
3sequential_5/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv1d_transpose_4/strided_slice/stack?
5sequential_5/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice/stack_1?
5sequential_5/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice/stack_2?
-sequential_5/conv1d_transpose_4/strided_sliceStridedSlice.sequential_5/conv1d_transpose_4/Shape:output:0<sequential_5/conv1d_transpose_4/strided_slice/stack:output:0>sequential_5/conv1d_transpose_4/strided_slice/stack_1:output:0>sequential_5/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv1d_transpose_4/strided_slice?
5sequential_5/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_4/strided_slice_1/stack?
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_1?
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_4/strided_slice_1/stack_2?
/sequential_5/conv1d_transpose_4/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_4/Shape:output:0>sequential_5/conv1d_transpose_4/strided_slice_1/stack:output:0@sequential_5/conv1d_transpose_4/strided_slice_1/stack_1:output:0@sequential_5/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv1d_transpose_4/strided_slice_1?
%sequential_5/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/conv1d_transpose_4/mul/y?
#sequential_5/conv1d_transpose_4/mulMul8sequential_5/conv1d_transpose_4/strided_slice_1:output:0.sequential_5/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_5/conv1d_transpose_4/mul?
'sequential_5/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv1d_transpose_4/stack/2?
%sequential_5/conv1d_transpose_4/stackPack6sequential_5/conv1d_transpose_4/strided_slice:output:0'sequential_5/conv1d_transpose_4/mul:z:00sequential_5/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_4/stack?
?sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
;sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims'sequential_5/reshape_5/Reshape:output:0Hsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims?
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_5_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
=sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
Dsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
>sequential_5/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice.sequential_5/conv1d_transpose_4/stack:output:0Msequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice?
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
@sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_4/stack:output:0Osequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1?
@sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1?
<sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis?
7sequential_5/conv1d_transpose_4/conv1d_transpose/concatConcatV2Gsequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0Isequential_5/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0Isequential_5/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0Esequential_5/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_5/conv1d_transpose_4/conv1d_transpose/concat?
0sequential_5/conv1d_transpose_4/conv1d_transposeConv2DBackpropInput@sequential_5/conv1d_transpose_4/conv1d_transpose/concat:output:0Fsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0Dsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_5/conv1d_transpose_4/conv1d_transpose?
8sequential_5/conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze9sequential_5/conv1d_transpose_4/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze?
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp?
'sequential_5/conv1d_transpose_4/BiasAddBiasAddAsequential_5/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0>sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_5/conv1d_transpose_4/BiasAdd?
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?
3sequential_5/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_16/batchnorm/add/y?
1sequential_5/batch_normalization_16/batchnorm/addAddV2Dsequential_5/batch_normalization_16/batchnorm/ReadVariableOp:value:0<sequential_5/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/add?
3sequential_5/batch_normalization_16/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_16/batchnorm/Rsqrt?
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_16/batchnorm/mulMul7sequential_5/batch_normalization_16/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/mul?
3sequential_5/batch_normalization_16/batchnorm/mul_1Mul0sequential_5/conv1d_transpose_4/BiasAdd:output:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_16/batchnorm/mul_1?
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_5_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1?
3sequential_5/batch_normalization_16/batchnorm/mul_2MulFsequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1:value:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_16/batchnorm/mul_2?
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_5_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2?
1sequential_5/batch_normalization_16/batchnorm/subSubFsequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2:value:07sequential_5/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_16/batchnorm/sub?
3sequential_5/batch_normalization_16/batchnorm/add_1AddV27sequential_5/batch_normalization_16/batchnorm/mul_1:z:05sequential_5/batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_16/batchnorm/add_1?
sequential_5/re_lu_7/ReluRelu7sequential_5/batch_normalization_16/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_5/re_lu_7/Relu?
%sequential_5/conv1d_transpose_5/ShapeShape'sequential_5/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_5/Shape?
3sequential_5/conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv1d_transpose_5/strided_slice/stack?
5sequential_5/conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice/stack_1?
5sequential_5/conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice/stack_2?
-sequential_5/conv1d_transpose_5/strided_sliceStridedSlice.sequential_5/conv1d_transpose_5/Shape:output:0<sequential_5/conv1d_transpose_5/strided_slice/stack:output:0>sequential_5/conv1d_transpose_5/strided_slice/stack_1:output:0>sequential_5/conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv1d_transpose_5/strided_slice?
5sequential_5/conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv1d_transpose_5/strided_slice_1/stack?
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_1?
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv1d_transpose_5/strided_slice_1/stack_2?
/sequential_5/conv1d_transpose_5/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_5/Shape:output:0>sequential_5/conv1d_transpose_5/strided_slice_1/stack:output:0@sequential_5/conv1d_transpose_5/strided_slice_1/stack_1:output:0@sequential_5/conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv1d_transpose_5/strided_slice_1?
%sequential_5/conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/conv1d_transpose_5/mul/y?
#sequential_5/conv1d_transpose_5/mulMul8sequential_5/conv1d_transpose_5/strided_slice_1:output:0.sequential_5/conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_5/conv1d_transpose_5/mul?
'sequential_5/conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv1d_transpose_5/stack/2?
%sequential_5/conv1d_transpose_5/stackPack6sequential_5/conv1d_transpose_5/strided_slice:output:0'sequential_5/conv1d_transpose_5/mul:z:00sequential_5/conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv1d_transpose_5/stack?
?sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim?
;sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDims'sequential_5/re_lu_7/Relu:activations:0Hsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims?
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_5_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim?
=sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1?
Dsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2?
>sequential_5/conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice.sequential_5/conv1d_transpose_5/stack:output:0Msequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice?
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack?
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2?
@sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice.sequential_5/conv1d_transpose_5/stack:output:0Osequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1?
@sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1?
<sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis?
7sequential_5/conv1d_transpose_5/conv1d_transpose/concatConcatV2Gsequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice:output:0Isequential_5/conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0Isequential_5/conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:0Esequential_5/conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_5/conv1d_transpose_5/conv1d_transpose/concat?
0sequential_5/conv1d_transpose_5/conv1d_transposeConv2DBackpropInput@sequential_5/conv1d_transpose_5/conv1d_transpose/concat:output:0Fsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:0Dsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_5/conv1d_transpose_5/conv1d_transpose?
8sequential_5/conv1d_transpose_5/conv1d_transpose/SqueezeSqueeze9sequential_5/conv1d_transpose_5/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze?
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp?
'sequential_5/conv1d_transpose_5/BiasAddBiasAddAsequential_5/conv1d_transpose_5/conv1d_transpose/Squeeze:output:0>sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_5/conv1d_transpose_5/BiasAdd?
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?
3sequential_5/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_17/batchnorm/add/y?
1sequential_5/batch_normalization_17/batchnorm/addAddV2Dsequential_5/batch_normalization_17/batchnorm/ReadVariableOp:value:0<sequential_5/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/add?
3sequential_5/batch_normalization_17/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_17/batchnorm/Rsqrt?
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_17/batchnorm/mulMul7sequential_5/batch_normalization_17/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/mul?
3sequential_5/batch_normalization_17/batchnorm/mul_1Mul0sequential_5/conv1d_transpose_5/BiasAdd:output:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_17/batchnorm/mul_1?
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_5_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1?
3sequential_5/batch_normalization_17/batchnorm/mul_2MulFsequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1:value:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_5/batch_normalization_17/batchnorm/mul_2?
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_5_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2?
1sequential_5/batch_normalization_17/batchnorm/subSubFsequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2:value:07sequential_5/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_5/batch_normalization_17/batchnorm/sub?
3sequential_5/batch_normalization_17/batchnorm/add_1AddV27sequential_5/batch_normalization_17/batchnorm/mul_1:z:05sequential_5/batch_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_5/batch_normalization_17/batchnorm/add_1?
sequential_5/re_lu_8/ReluRelu7sequential_5/batch_normalization_17/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_5/re_lu_8/Relu?
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_5/flatten_5/Const?
sequential_5/flatten_5/ReshapeReshape'sequential_5/re_lu_8/Relu:activations:0%sequential_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_5/flatten_5/Reshape?
+sequential_5/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_5/dense_11/MatMul/ReadVariableOp?
sequential_5/dense_11/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_11/MatMul?
sequential_5/dense_11/TanhTanh&sequential_5/dense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_11/Tanh?
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_4/dense_8/MatMul/ReadVariableOp?
sequential_4/dense_8/MatMulMatMulsequential_5/dense_11/Tanh:y:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_8/MatMul?
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?
3sequential_4/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_12/batchnorm/add/y?
1sequential_4/batch_normalization_12/batchnorm/addAddV2Dsequential_4/batch_normalization_12/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/add?
3sequential_4/batch_normalization_12/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_12/batchnorm/Rsqrt?
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_12/batchnorm/mulMul7sequential_4/batch_normalization_12/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/mul?
3sequential_4/batch_normalization_12/batchnorm/mul_1Mul%sequential_4/dense_8/MatMul:product:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????25
3sequential_4/batch_normalization_12/batchnorm/mul_1?
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?
3sequential_4/batch_normalization_12/batchnorm/mul_2MulFsequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_12/batchnorm/mul_2?
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2?
1sequential_4/batch_normalization_12/batchnorm/subSubFsequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_12/batchnorm/sub?
3sequential_4/batch_normalization_12/batchnorm/add_1AddV27sequential_4/batch_normalization_12/batchnorm/mul_1:z:05sequential_4/batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????25
3sequential_4/batch_normalization_12/batchnorm/add_1?
$sequential_4/leaky_re_lu_6/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_12/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_6/LeakyRelu?
sequential_4/reshape_4/ShapeShape2sequential_4/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_4/reshape_4/Shape?
*sequential_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_4/strided_slice/stack?
,sequential_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_4/strided_slice/stack_1?
,sequential_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_4/strided_slice/stack_2?
$sequential_4/reshape_4/strided_sliceStridedSlice%sequential_4/reshape_4/Shape:output:03sequential_4/reshape_4/strided_slice/stack:output:05sequential_4/reshape_4/strided_slice/stack_1:output:05sequential_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_4/strided_slice?
&sequential_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_4/Reshape/shape/1?
&sequential_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_4/Reshape/shape/2?
$sequential_4/reshape_4/Reshape/shapePack-sequential_4/reshape_4/strided_slice:output:0/sequential_4/reshape_4/Reshape/shape/1:output:0/sequential_4/reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_4/Reshape/shape?
sequential_4/reshape_4/ReshapeReshape2sequential_4/leaky_re_lu_6/LeakyRelu:activations:0-sequential_4/reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_4/reshape_4/Reshape?
+sequential_4/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_4/conv1d_4/conv1d/ExpandDims/dim?
'sequential_4/conv1d_4/conv1d/ExpandDims
ExpandDims'sequential_4/reshape_4/Reshape:output:04sequential_4/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_4/conv1d_4/conv1d/ExpandDims?
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_4_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_4/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_4/conv1d_4/conv1d/ExpandDims_1/dim?
)sequential_4/conv1d_4/conv1d/ExpandDims_1
ExpandDims@sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_4/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_4/conv1d_4/conv1d/ExpandDims_1?
sequential_4/conv1d_4/conv1dConv2D0sequential_4/conv1d_4/conv1d/ExpandDims:output:02sequential_4/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_4/conv1d_4/conv1d?
$sequential_4/conv1d_4/conv1d/SqueezeSqueeze%sequential_4/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_4/conv1d_4/conv1d/Squeeze?
,sequential_4/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/conv1d_4/BiasAdd/ReadVariableOp?
sequential_4/conv1d_4/BiasAddBiasAdd-sequential_4/conv1d_4/conv1d/Squeeze:output:04sequential_4/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_4/conv1d_4/BiasAdd?
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?
3sequential_4/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_13/batchnorm/add/y?
1sequential_4/batch_normalization_13/batchnorm/addAddV2Dsequential_4/batch_normalization_13/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/add?
3sequential_4/batch_normalization_13/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_13/batchnorm/Rsqrt?
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_13/batchnorm/mulMul7sequential_4/batch_normalization_13/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/mul?
3sequential_4/batch_normalization_13/batchnorm/mul_1Mul&sequential_4/conv1d_4/BiasAdd:output:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_13/batchnorm/mul_1?
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?
3sequential_4/batch_normalization_13/batchnorm/mul_2MulFsequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_13/batchnorm/mul_2?
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2?
1sequential_4/batch_normalization_13/batchnorm/subSubFsequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_13/batchnorm/sub?
3sequential_4/batch_normalization_13/batchnorm/add_1AddV27sequential_4/batch_normalization_13/batchnorm/mul_1:z:05sequential_4/batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_13/batchnorm/add_1?
$sequential_4/leaky_re_lu_7/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_13/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_7/LeakyRelu?
+sequential_4/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_4/conv1d_5/conv1d/ExpandDims/dim?
'sequential_4/conv1d_5/conv1d/ExpandDims
ExpandDims2sequential_4/leaky_re_lu_7/LeakyRelu:activations:04sequential_4/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_4/conv1d_5/conv1d/ExpandDims?
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_4_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_4/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_4/conv1d_5/conv1d/ExpandDims_1/dim?
)sequential_4/conv1d_5/conv1d/ExpandDims_1
ExpandDims@sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_4/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_4/conv1d_5/conv1d/ExpandDims_1?
sequential_4/conv1d_5/conv1dConv2D0sequential_4/conv1d_5/conv1d/ExpandDims:output:02sequential_4/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_4/conv1d_5/conv1d?
$sequential_4/conv1d_5/conv1d/SqueezeSqueeze%sequential_4/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_4/conv1d_5/conv1d/Squeeze?
,sequential_4/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/conv1d_5/BiasAdd/ReadVariableOp?
sequential_4/conv1d_5/BiasAddBiasAdd-sequential_4/conv1d_5/conv1d/Squeeze:output:04sequential_4/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_4/conv1d_5/BiasAdd?
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?
3sequential_4/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_4/batch_normalization_14/batchnorm/add/y?
1sequential_4/batch_normalization_14/batchnorm/addAddV2Dsequential_4/batch_normalization_14/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/add?
3sequential_4/batch_normalization_14/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_14/batchnorm/Rsqrt?
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp?
1sequential_4/batch_normalization_14/batchnorm/mulMul7sequential_4/batch_normalization_14/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/mul?
3sequential_4/batch_normalization_14/batchnorm/mul_1Mul&sequential_4/conv1d_5/BiasAdd:output:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_14/batchnorm/mul_1?
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_14_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?
3sequential_4/batch_normalization_14/batchnorm/mul_2MulFsequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_4/batch_normalization_14/batchnorm/mul_2?
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_14_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2?
1sequential_4/batch_normalization_14/batchnorm/subSubFsequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_4/batch_normalization_14/batchnorm/sub?
3sequential_4/batch_normalization_14/batchnorm/add_1AddV27sequential_4/batch_normalization_14/batchnorm/mul_1:z:05sequential_4/batch_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_4/batch_normalization_14/batchnorm/add_1?
$sequential_4/leaky_re_lu_8/LeakyRelu	LeakyRelu7sequential_4/batch_normalization_14/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_4/leaky_re_lu_8/LeakyRelu?
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_4/flatten_4/Const?
sequential_4/flatten_4/ReshapeReshape2sequential_4/leaky_re_lu_8/LeakyRelu:activations:0%sequential_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_4/flatten_4/Reshape?
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_4/dense_9/MatMul/ReadVariableOp?
sequential_4/dense_9/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_9/MatMul?
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_9/BiasAdd/ReadVariableOp?
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_9/BiasAdd?
IdentityIdentity%sequential_4/dense_9/BiasAdd:output:0=^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp-^sequential_4/conv1d_4/BiasAdd/ReadVariableOp9^sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp-^sequential_4/conv1d_5/BiasAdd/ReadVariableOp9^sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOp,^sequential_4/dense_9/BiasAdd/ReadVariableOp+^sequential_4/dense_9/MatMul/ReadVariableOp=^sequential_5/batch_normalization_15/batchnorm/ReadVariableOp?^sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1?^sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2A^sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp=^sequential_5/batch_normalization_16/batchnorm/ReadVariableOp?^sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1?^sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2A^sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp=^sequential_5/batch_normalization_17/batchnorm/ReadVariableOp?^sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1?^sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2A^sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp7^sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOpM^sequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp7^sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOpM^sequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp,^sequential_5/dense_10/MatMul/ReadVariableOp,^sequential_5/dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2|
<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp<sequential_4/batch_normalization_12/batchnorm/ReadVariableOp2?
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_12?
>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_12/batchnorm/ReadVariableOp_22?
@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_12/batchnorm/mul/ReadVariableOp2|
<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp<sequential_4/batch_normalization_13/batchnorm/ReadVariableOp2?
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_12?
>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_13/batchnorm/ReadVariableOp_22?
@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_13/batchnorm/mul/ReadVariableOp2|
<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp<sequential_4/batch_normalization_14/batchnorm/ReadVariableOp2?
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_12?
>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_14/batchnorm/ReadVariableOp_22?
@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_14/batchnorm/mul/ReadVariableOp2\
,sequential_4/conv1d_4/BiasAdd/ReadVariableOp,sequential_4/conv1d_4/BiasAdd/ReadVariableOp2t
8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp8sequential_4/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2\
,sequential_4/conv1d_5/BiasAdd/ReadVariableOp,sequential_4/conv1d_5/BiasAdd/ReadVariableOp2t
8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp8sequential_4/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2Z
+sequential_4/dense_9/BiasAdd/ReadVariableOp+sequential_4/dense_9/BiasAdd/ReadVariableOp2X
*sequential_4/dense_9/MatMul/ReadVariableOp*sequential_4/dense_9/MatMul/ReadVariableOp2|
<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp<sequential_5/batch_normalization_15/batchnorm/ReadVariableOp2?
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_1>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_12?
>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_2>sequential_5/batch_normalization_15/batchnorm/ReadVariableOp_22?
@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_15/batchnorm/mul/ReadVariableOp2|
<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp<sequential_5/batch_normalization_16/batchnorm/ReadVariableOp2?
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_1>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_12?
>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_2>sequential_5/batch_normalization_16/batchnorm/ReadVariableOp_22?
@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_16/batchnorm/mul/ReadVariableOp2|
<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp<sequential_5/batch_normalization_17/batchnorm/ReadVariableOp2?
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_1>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_12?
>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_2>sequential_5/batch_normalization_17/batchnorm/ReadVariableOp_22?
@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_17/batchnorm/mul/ReadVariableOp2p
6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp6sequential_5/conv1d_transpose_4/BiasAdd/ReadVariableOp2?
Lsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_5/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2p
6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp6sequential_5/conv1d_transpose_5/BiasAdd/ReadVariableOp2?
Lsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_5/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2Z
+sequential_5/dense_10/MatMul/ReadVariableOp+sequential_5/dense_10/MatMul/ReadVariableOp2Z
+sequential_5/dense_11/MatMul/ReadVariableOp+sequential_5/dense_11/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_16869

inputs+
'dense_10_matmul_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resource@
<batch_normalization_15_batchnorm_mul_readvariableop_resource>
:batch_normalization_15_batchnorm_readvariableop_1_resource>
:batch_normalization_15_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_4_biasadd_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource@
<batch_normalization_16_batchnorm_mul_readvariableop_resource>
:batch_normalization_16_batchnorm_readvariableop_1_resource>
:batch_normalization_16_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_5_biasadd_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource@
<batch_normalization_17_batchnorm_mul_readvariableop_resource>
:batch_normalization_17_batchnorm_readvariableop_1_resource>
:batch_normalization_17_batchnorm_readvariableop_2_resource+
'dense_11_matmul_readvariableop_resource
identity??/batch_normalization_15/batchnorm/ReadVariableOp?1batch_normalization_15/batchnorm/ReadVariableOp_1?1batch_normalization_15/batchnorm/ReadVariableOp_2?3batch_normalization_15/batchnorm/mul/ReadVariableOp?/batch_normalization_16/batchnorm/ReadVariableOp?1batch_normalization_16/batchnorm/ReadVariableOp_1?1batch_normalization_16/batchnorm/ReadVariableOp_2?3batch_normalization_16/batchnorm/mul/ReadVariableOp?/batch_normalization_17/batchnorm/ReadVariableOp?1batch_normalization_17/batchnorm/ReadVariableOp_1?1batch_normalization_17/batchnorm/ReadVariableOp_2?3batch_normalization_17/batchnorm/mul/ReadVariableOp?)conv1d_transpose_4/BiasAdd/ReadVariableOp??conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_5/BiasAdd/ReadVariableOp??conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMul?
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp?
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_15/batchnorm/add/y?
$batch_normalization_15/batchnorm/addAddV27batch_normalization_15/batchnorm/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add?
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt?
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp?
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul?
&batch_normalization_15/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/mul_1?
1batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_1?
&batch_normalization_15/batchnorm/mul_2Mul9batch_normalization_15/batchnorm/ReadVariableOp_1:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2?
1batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_2?
$batch_normalization_15/batchnorm/subSub9batch_normalization_15/batchnorm/ReadVariableOp_2:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub?
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/add_1?
re_lu_6/ReluRelu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_6/Relul
reshape_5/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
reshape_5/Shape?
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_5/strided_slice/stack?
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_1?
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_2?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_5/strided_slicex
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/1x
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/2?
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_5/Reshape/shape?
reshape_5/ReshapeReshapere_lu_6/Relu:activations:0 reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_5/Reshape~
conv1d_transpose_4/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape?
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack?
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1?
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2?
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice?
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack?
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1?
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2?
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y?
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2?
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stack?
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDimsreshape_5/Reshape:output:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_4/conv1d_transpose/ExpandDims?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_slice?
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1?
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1?
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis?
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concat?
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transpose?
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze?
)conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_4/BiasAdd/ReadVariableOp?
conv1d_transpose_4/BiasAddBiasAdd4conv1d_transpose_4/conv1d_transpose/Squeeze:output:01conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_4/BiasAdd?
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp?
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_16/batchnorm/add/y?
$batch_normalization_16/batchnorm/addAddV27batch_normalization_16/batchnorm/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add?
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt?
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp?
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul?
&batch_normalization_16/batchnorm/mul_1Mul#conv1d_transpose_4/BiasAdd:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1?
1batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_1?
&batch_normalization_16/batchnorm/mul_2Mul9batch_normalization_16/batchnorm/ReadVariableOp_1:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2?
1batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_2?
$batch_normalization_16/batchnorm/subSub9batch_normalization_16/batchnorm/ReadVariableOp_2:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub?
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1?
re_lu_7/ReluRelu*batch_normalization_16/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_7/Relu~
conv1d_transpose_5/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_5/Shape?
&conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_5/strided_slice/stack?
(conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice/stack_1?
(conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice/stack_2?
 conv1d_transpose_5/strided_sliceStridedSlice!conv1d_transpose_5/Shape:output:0/conv1d_transpose_5/strided_slice/stack:output:01conv1d_transpose_5/strided_slice/stack_1:output:01conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_5/strided_slice?
(conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_5/strided_slice_1/stack?
*conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_5/strided_slice_1/stack_1?
*conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_5/strided_slice_1/stack_2?
"conv1d_transpose_5/strided_slice_1StridedSlice!conv1d_transpose_5/Shape:output:01conv1d_transpose_5/strided_slice_1/stack:output:03conv1d_transpose_5/strided_slice_1/stack_1:output:03conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_5/strided_slice_1v
conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_5/mul/y?
conv1d_transpose_5/mulMul+conv1d_transpose_5/strided_slice_1:output:0!conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_5/mulz
conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_5/stack/2?
conv1d_transpose_5/stackPack)conv1d_transpose_5/strided_slice:output:0conv1d_transpose_5/mul:z:0#conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_5/stack?
2conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_5/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDimsre_lu_7/Relu:activations:0;conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_5/conv1d_transpose/ExpandDims?
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_5/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_5/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_5/stack:output:0@conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_5/conv1d_transpose/strided_slice?
9conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_5/stack:output:0Bconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_5/conv1d_transpose/strided_slice_1?
3conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_5/conv1d_transpose/concat/values_1?
/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_5/conv1d_transpose/concat/axis?
*conv1d_transpose_5/conv1d_transpose/concatConcatV2:conv1d_transpose_5/conv1d_transpose/strided_slice:output:0<conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_5/conv1d_transpose/concat?
#conv1d_transpose_5/conv1d_transposeConv2DBackpropInput3conv1d_transpose_5/conv1d_transpose/concat:output:09conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_5/conv1d_transpose?
+conv1d_transpose_5/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_5/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_5/conv1d_transpose/Squeeze?
)conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_5/BiasAdd/ReadVariableOp?
conv1d_transpose_5/BiasAddBiasAdd4conv1d_transpose_5/conv1d_transpose/Squeeze:output:01conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_5/BiasAdd?
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp?
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_17/batchnorm/add/y?
$batch_normalization_17/batchnorm/addAddV27batch_normalization_17/batchnorm/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/add?
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/Rsqrt?
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp?
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/mul?
&batch_normalization_17/batchnorm/mul_1Mul#conv1d_transpose_5/BiasAdd:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/mul_1?
1batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_1?
&batch_normalization_17/batchnorm/mul_2Mul9batch_normalization_17/batchnorm/ReadVariableOp_1:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/mul_2?
1batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_2?
$batch_normalization_17/batchnorm/subSub9batch_normalization_17/batchnorm/ReadVariableOp_2:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/sub?
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/add_1?
re_lu_8/ReluRelu*batch_normalization_17/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_8/Relus
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshapere_lu_8/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshape?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_5/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanh?
IdentityIdentitydense_11/Tanh:y:00^batch_normalization_15/batchnorm/ReadVariableOp2^batch_normalization_15/batchnorm/ReadVariableOp_12^batch_normalization_15/batchnorm/ReadVariableOp_24^batch_normalization_15/batchnorm/mul/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp2^batch_normalization_16/batchnorm/ReadVariableOp_12^batch_normalization_16/batchnorm/ReadVariableOp_24^batch_normalization_16/batchnorm/mul/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp2^batch_normalization_17/batchnorm/ReadVariableOp_12^batch_normalization_17/batchnorm/ReadVariableOp_24^batch_normalization_17/batchnorm/mul/ReadVariableOp*^conv1d_transpose_4/BiasAdd/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_5/BiasAdd/ReadVariableOp@^conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2f
1batch_normalization_15/batchnorm/ReadVariableOp_11batch_normalization_15/batchnorm/ReadVariableOp_12f
1batch_normalization_15/batchnorm/ReadVariableOp_21batch_normalization_15/batchnorm/ReadVariableOp_22j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2f
1batch_normalization_16/batchnorm/ReadVariableOp_11batch_normalization_16/batchnorm/ReadVariableOp_12f
1batch_normalization_16/batchnorm/ReadVariableOp_21batch_normalization_16/batchnorm/ReadVariableOp_22j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2f
1batch_normalization_17/batchnorm/ReadVariableOp_11batch_normalization_17/batchnorm/ReadVariableOp_12f
1batch_normalization_17/batchnorm/ReadVariableOp_21batch_normalization_17/batchnorm/ReadVariableOp_22j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2V
)conv1d_transpose_4/BiasAdd/ReadVariableOp)conv1d_transpose_4/BiasAdd/ReadVariableOp2?
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_5/BiasAdd/ReadVariableOp)conv1d_transpose_5/BiasAdd/ReadVariableOp2?
?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_6_layer_call_fn_17385

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
GPU 2J 8? *K
fFRD
B__inference_re_lu_6_layer_call_and_return_conditional_losses_135932
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
?
?
,__inference_sequential_4_layer_call_fn_17236

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

unknown_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_149422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_5_layer_call_fn_17403

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
GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_136142
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
?
^
B__inference_re_lu_7_layer_call_and_return_conditional_losses_13667

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
?

?
,__inference_sequential_5_layer_call_fn_16951

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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_139692
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
?
d
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_14782

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???>2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18103

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
(__inference_dense_11_layer_call_fn_17619

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
C__inference_dense_11_layer_call_and_return_conditional_losses_137562
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
?/
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_14104

inputs
assignmovingavg_14079
assignmovingavg_1_14085)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14079*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_14079*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14079*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/14079*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_14079AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/14079*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14085*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_14085*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14085*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/14085*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_14085AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/14085*
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
I
-__inference_leaky_re_lu_7_layer_call_fn_17941

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146472
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_13139

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
?
6__inference_batch_normalization_15_layer_call_fn_17375

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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_131392
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
?
d
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_18134

inputs
identityh
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????*
alpha%???>2
	LeakyReluo
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15329
sequential_5_input
sequential_5_15166
sequential_5_15168
sequential_5_15170
sequential_5_15172
sequential_5_15174
sequential_5_15176
sequential_5_15178
sequential_5_15180
sequential_5_15182
sequential_5_15184
sequential_5_15186
sequential_5_15188
sequential_5_15190
sequential_5_15192
sequential_5_15194
sequential_5_15196
sequential_5_15198
sequential_5_15200
sequential_4_15289
sequential_4_15291
sequential_4_15293
sequential_4_15295
sequential_4_15297
sequential_4_15299
sequential_4_15301
sequential_4_15303
sequential_4_15305
sequential_4_15307
sequential_4_15309
sequential_4_15311
sequential_4_15313
sequential_4_15315
sequential_4_15317
sequential_4_15319
sequential_4_15321
sequential_4_15323
sequential_4_15325
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputsequential_5_15166sequential_5_15168sequential_5_15170sequential_5_15172sequential_5_15174sequential_5_15176sequential_5_15178sequential_5_15180sequential_5_15182sequential_5_15184sequential_5_15186sequential_5_15188sequential_5_15190sequential_5_15192sequential_5_15194sequential_5_15196sequential_5_15198sequential_5_15200*
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
GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_138762&
$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0sequential_4_15289sequential_4_15291sequential_4_15293sequential_4_15295sequential_4_15297sequential_4_15299sequential_4_15301sequential_4_15303sequential_4_15305sequential_4_15307sequential_4_15309sequential_4_15311sequential_4_15313sequential_4_15315sequential_4_15317sequential_4_15319sequential_4_15321sequential_4_15323sequential_4_15325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_149422&
$sequential_4/StatefulPartitionedCall?
IdentityIdentity-sequential_4/StatefulPartitionedCall:output:0%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
?
?
6__inference_batch_normalization_14_layer_call_fn_18116

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
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_147212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_conv1d_5_layer_call_fn_17965

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
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_146702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18001

inputs
assignmovingavg_17976
assignmovingavg_1_17982)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17976*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_17976*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17976*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/17976*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_17976AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/17976*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17982*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_17982*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17982*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/17982*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_17982AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/17982*
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
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_14606

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv1d_transpose_4_layer_call_fn_13200

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
GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_131902
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
?
?
C__inference_dense_11_layer_call_and_return_conditional_losses_17612

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
?
?
,__inference_sequential_6_layer_call_fn_15728
sequential_5_input
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_156512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_5_input
?
^
B__inference_re_lu_8_layer_call_and_return_conditional_losses_13720

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
?
?
C__inference_dense_11_layer_call_and_return_conditional_losses_13756

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
?
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_14491

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
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
?
6__inference_batch_normalization_14_layer_call_fn_18034

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
GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_143842
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
?
I
-__inference_leaky_re_lu_6_layer_call_fn_17725

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
GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_144912
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
sequential_5_input;
$serving_default_sequential_5_input:0?????????@
sequential_40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ե
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"ǣ
_tf_keras_sequential??{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_5_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_5_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}}
?V
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer-9
layer-10
layer_with_weights-6
layer-11
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?R
_tf_keras_sequential?R{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?T
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
 layer_with_weights-5
 layer-8
!layer-9
"layer-10
#layer_with_weights-6
#layer-11
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?P
_tf_keras_sequential?P{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
 "
trackable_list_wrapper
?
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
>22
?23
@24"
trackable_list_wrapper
?
(0
)1
*2
A3
B4
+5
,6
-7
.8
C9
D10
/11
012
113
214
E15
F16
317
418
519
620
G21
H22
723
824
925
:26
I27
J28
;29
<30
=31
>32
K33
L34
?35
@36"
trackable_list_wrapper
?
Mlayer_regularization_losses

Nlayers
Olayer_metrics
regularization_losses
Pnon_trainable_variables
trainable_variables
Qmetrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

(kernel
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?	
Vaxis
	)gamma
*beta
Amoving_mean
Bmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
[regularization_losses
\trainable_variables
]	variables
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
_regularization_losses
`trainable_variables
a	variables
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?


+kernel
,bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
?	
gaxis
	-gamma
.beta
Cmoving_mean
Dmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


/kernel
0bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?	
taxis
	1gamma
2beta
Emoving_mean
Fmoving_variance
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 6]}}
?
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
}regularization_losses
~trainable_variables
	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
 "
trackable_list_wrapper
v
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
?
(0
)1
*2
A3
B4
+5
,6
-7
.8
C9
D10
/11
012
113
214
E15
F16
317"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
regularization_losses
?non_trainable_variables
trainable_variables
?metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

4kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?	
	?axis
	5gamma
6beta
Gmoving_mean
Hmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?	

7kernel
8bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
?	
	?axis
	9gamma
:beta
Imoving_mean
Jmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

;kernel
<bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?	
	?axis
	=gamma
>beta
Kmoving_mean
Lmoving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 6]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

?kernel
@bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
 "
trackable_list_wrapper
~
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12"
trackable_list_wrapper
?
40
51
62
G3
H4
75
86
97
:8
I9
J10
;11
<12
=13
>14
K15
L16
?17
@18"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
$regularization_losses
?non_trainable_variables
%trainable_variables
?metrics
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_10/kernel
*:(2batch_normalization_15/gamma
):'2batch_normalization_15/beta
/:-2conv1d_transpose_4/kernel
%:#2conv1d_transpose_4/bias
*:(2batch_normalization_16/gamma
):'2batch_normalization_16/beta
/:-2conv1d_transpose_5/kernel
%:#2conv1d_transpose_5/bias
*:(2batch_normalization_17/gamma
):'2batch_normalization_17/beta
!:2dense_11/kernel
 :2dense_8/kernel
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
%:#2conv1d_4/kernel
:2conv1d_4/bias
*:(2batch_normalization_13/gamma
):'2batch_normalization_13/beta
%:#2conv1d_5/kernel
:2conv1d_5/bias
*:(2batch_normalization_14/gamma
):'2batch_normalization_14/beta
 :2dense_9/kernel
:2dense_9/bias
2:0 (2"batch_normalization_15/moving_mean
6:4 (2&batch_normalization_15/moving_variance
2:0 (2"batch_normalization_16/moving_mean
6:4 (2&batch_normalization_16/moving_variance
2:0 (2"batch_normalization_17/moving_mean
6:4 (2&batch_normalization_17/moving_variance
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
2:0 (2"batch_normalization_13/moving_mean
6:4 (2&batch_normalization_13/moving_variance
2:0 (2"batch_normalization_14/moving_mean
6:4 (2&batch_normalization_14/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
v
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
Rregularization_losses
?non_trainable_variables
Strainable_variables
?metrics
T	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
<
)0
*1
A2
B3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
Wregularization_losses
?non_trainable_variables
Xtrainable_variables
?metrics
Y	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
[regularization_losses
?non_trainable_variables
\trainable_variables
?metrics
]	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
_regularization_losses
?non_trainable_variables
`trainable_variables
?metrics
a	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
cregularization_losses
?non_trainable_variables
dtrainable_variables
?metrics
e	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
<
-0
.1
C2
D3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
hregularization_losses
?non_trainable_variables
itrainable_variables
?metrics
j	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
lregularization_losses
?non_trainable_variables
mtrainable_variables
?metrics
n	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
pregularization_losses
?non_trainable_variables
qtrainable_variables
?metrics
r	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
<
10
21
E2
F3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
uregularization_losses
?non_trainable_variables
vtrainable_variables
?metrics
w	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
yregularization_losses
?non_trainable_variables
ztrainable_variables
?metrics
{	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
}regularization_losses
?non_trainable_variables
~trainable_variables
?metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
v
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
A0
B1
C2
D3
E4
F5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
<
50
61
G2
H3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
<
90
:1
I2
J3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
<
=0
>1
K2
L3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?layer_metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
 8
!9
"10
#11"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
G0
H1
I2
J3
K4
L5"
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
A0
B1"
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
.
C0
D1"
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
E0
F1"
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
.
G0
H1"
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
.
I0
J1"
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
K0
L1"
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
?2?
,__inference_sequential_6_layer_call_fn_15728
,__inference_sequential_6_layer_call_fn_15569
,__inference_sequential_6_layer_call_fn_16456
,__inference_sequential_6_layer_call_fn_16535?
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
 __inference__wrapped_model_13010?
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
annotations? *1?.
,?)
sequential_5_input?????????
?2?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15329
G__inference_sequential_6_layer_call_and_return_conditional_losses_15409
G__inference_sequential_6_layer_call_and_return_conditional_losses_16377
G__inference_sequential_6_layer_call_and_return_conditional_losses_16141?
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
?2?
,__inference_sequential_5_layer_call_fn_16951
,__inference_sequential_5_layer_call_fn_16910
,__inference_sequential_5_layer_call_fn_14008
,__inference_sequential_5_layer_call_fn_13915?
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_13821
G__inference_sequential_5_layer_call_and_return_conditional_losses_16869
G__inference_sequential_5_layer_call_and_return_conditional_losses_16726
G__inference_sequential_5_layer_call_and_return_conditional_losses_13769?
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
?2?
,__inference_sequential_4_layer_call_fn_17236
,__inference_sequential_4_layer_call_fn_17279
,__inference_sequential_4_layer_call_fn_14983
,__inference_sequential_4_layer_call_fn_15080?
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
G__inference_sequential_4_layer_call_and_return_conditional_losses_14885
G__inference_sequential_4_layer_call_and_return_conditional_losses_17096
G__inference_sequential_4_layer_call_and_return_conditional_losses_17193
G__inference_sequential_4_layer_call_and_return_conditional_losses_14831?
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
?B?
#__inference_signature_wrapper_15809sequential_5_input"?
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
 
?2?
(__inference_dense_10_layer_call_fn_17293?
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
C__inference_dense_10_layer_call_and_return_conditional_losses_17286?
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
6__inference_batch_normalization_15_layer_call_fn_17375
6__inference_batch_normalization_15_layer_call_fn_17362?
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
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17329
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17349?
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
'__inference_re_lu_6_layer_call_fn_17385?
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
B__inference_re_lu_6_layer_call_and_return_conditional_losses_17380?
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
)__inference_reshape_5_layer_call_fn_17403?
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
D__inference_reshape_5_layer_call_and_return_conditional_losses_17398?
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
2__inference_conv1d_transpose_4_layer_call_fn_13200?
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
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_13190?
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
6__inference_batch_normalization_16_layer_call_fn_17485
6__inference_batch_normalization_16_layer_call_fn_17472?
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
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17459
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17439?
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
'__inference_re_lu_7_layer_call_fn_17495?
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
B__inference_re_lu_7_layer_call_and_return_conditional_losses_17490?
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
2__inference_conv1d_transpose_5_layer_call_fn_13390?
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
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_13380?
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
6__inference_batch_normalization_17_layer_call_fn_17577
6__inference_batch_normalization_17_layer_call_fn_17564?
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
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17531
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17551?
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
'__inference_re_lu_8_layer_call_fn_17587?
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
B__inference_re_lu_8_layer_call_and_return_conditional_losses_17582?
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
)__inference_flatten_5_layer_call_fn_17604?
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
D__inference_flatten_5_layer_call_and_return_conditional_losses_17599?
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
(__inference_dense_11_layer_call_fn_17619?
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
C__inference_dense_11_layer_call_and_return_conditional_losses_17612?
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
'__inference_dense_8_layer_call_fn_17633?
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
B__inference_dense_8_layer_call_and_return_conditional_losses_17626?
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
6__inference_batch_normalization_12_layer_call_fn_17715
6__inference_batch_normalization_12_layer_call_fn_17702?
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
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17669
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17689?
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
-__inference_leaky_re_lu_6_layer_call_fn_17725?
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
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_17720?
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
)__inference_reshape_4_layer_call_fn_17743?
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
D__inference_reshape_4_layer_call_and_return_conditional_losses_17738?
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
(__inference_conv1d_4_layer_call_fn_17767?
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
C__inference_conv1d_4_layer_call_and_return_conditional_losses_17758?
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
?2?
6__inference_batch_normalization_13_layer_call_fn_17931
6__inference_batch_normalization_13_layer_call_fn_17836
6__inference_batch_normalization_13_layer_call_fn_17918
6__inference_batch_normalization_13_layer_call_fn_17849?
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
?2?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17823
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17885
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17905
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17803?
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
-__inference_leaky_re_lu_7_layer_call_fn_17941?
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
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_17936?
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
(__inference_conv1d_5_layer_call_fn_17965?
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
C__inference_conv1d_5_layer_call_and_return_conditional_losses_17956?
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
?2?
6__inference_batch_normalization_14_layer_call_fn_18129
6__inference_batch_normalization_14_layer_call_fn_18116
6__inference_batch_normalization_14_layer_call_fn_18034
6__inference_batch_normalization_14_layer_call_fn_18047?
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
?2?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18021
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18001
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18083
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18103?
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
-__inference_leaky_re_lu_8_layer_call_fn_18139?
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
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_18134?
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
)__inference_flatten_4_layer_call_fn_18150?
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_18145?
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
'__inference_dense_9_layer_call_fn_18169?
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
B__inference_dense_9_layer_call_and_return_conditional_losses_18160?
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
 ?
 __inference__wrapped_model_13010?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@;?8
1?.
,?)
sequential_5_input?????????
? ";?8
6
sequential_4&?#
sequential_4??????????
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17669bGH563?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_17689bH5G63?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
6__inference_batch_normalization_12_layer_call_fn_17702UGH563?0
)?&
 ?
inputs?????????
p
? "???????????
6__inference_batch_normalization_12_layer_call_fn_17715UH5G63?0
)?&
 ?
inputs?????????
p 
? "???????????
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17803jIJ9:7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17823jJ9I:7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17885|IJ9:@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_17905|J9I:@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
6__inference_batch_normalization_13_layer_call_fn_17836]IJ9:7?4
-?*
$?!
inputs?????????
p
? "???????????
6__inference_batch_normalization_13_layer_call_fn_17849]J9I:7?4
-?*
$?!
inputs?????????
p 
? "???????????
6__inference_batch_normalization_13_layer_call_fn_17918oIJ9:@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_13_layer_call_fn_17931oJ9I:@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18001|KL=>@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18021|L=K>@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18083jKL=>7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_18103jL=K>7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
6__inference_batch_normalization_14_layer_call_fn_18034oKL=>@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_14_layer_call_fn_18047oL=K>@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
6__inference_batch_normalization_14_layer_call_fn_18116]KL=>7?4
-?*
$?!
inputs?????????
p
? "???????????
6__inference_batch_normalization_14_layer_call_fn_18129]L=K>7?4
-?*
$?!
inputs?????????
p 
? "???????????
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17329bAB)*3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17349bB)A*3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
6__inference_batch_normalization_15_layer_call_fn_17362UAB)*3?0
)?&
 ?
inputs?????????
p
? "???????????
6__inference_batch_normalization_15_layer_call_fn_17375UB)A*3?0
)?&
 ?
inputs?????????
p 
? "???????????
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17439|CD-.@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17459|D-C.@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
6__inference_batch_normalization_16_layer_call_fn_17472oCD-.@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_16_layer_call_fn_17485oD-C.@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17531|EF12@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17551|F1E2@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
6__inference_batch_normalization_17_layer_call_fn_17564oEF12@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_17_layer_call_fn_17577oF1E2@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
C__inference_conv1d_4_layer_call_and_return_conditional_losses_17758d783?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
(__inference_conv1d_4_layer_call_fn_17767W783?0
)?&
$?!
inputs?????????
? "???????????
C__inference_conv1d_5_layer_call_and_return_conditional_losses_17956d;<3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
(__inference_conv1d_5_layer_call_fn_17965W;<3?0
)?&
$?!
inputs?????????
? "???????????
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_13190v+,<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_4_layer_call_fn_13200i+,<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_13380v/0<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_5_layer_call_fn_13390i/0<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
C__inference_dense_10_layer_call_and_return_conditional_losses_17286[(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
(__inference_dense_10_layer_call_fn_17293N(/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_11_layer_call_and_return_conditional_losses_17612d38?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
(__inference_dense_11_layer_call_fn_17619W38?5
.?+
)?&
inputs??????????????????
? "???????????
B__inference_dense_8_layer_call_and_return_conditional_losses_17626[4/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
'__inference_dense_8_layer_call_fn_17633N4/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_9_layer_call_and_return_conditional_losses_18160\?@/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_9_layer_call_fn_18169O?@/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_18145\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? |
)__inference_flatten_4_layer_call_fn_18150O3?0
)?&
$?!
inputs?????????
? "???????????
D__inference_flatten_5_layer_call_and_return_conditional_losses_17599n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
)__inference_flatten_5_layer_call_fn_17604a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_17720X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_leaky_re_lu_6_layer_call_fn_17725K/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_17936`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
-__inference_leaky_re_lu_7_layer_call_fn_17941S3?0
)?&
$?!
inputs?????????
? "???????????
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_18134`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
-__inference_leaky_re_lu_8_layer_call_fn_18139S3?0
)?&
$?!
inputs?????????
? "???????????
B__inference_re_lu_6_layer_call_and_return_conditional_losses_17380X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
'__inference_re_lu_6_layer_call_fn_17385K/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_re_lu_7_layer_call_and_return_conditional_losses_17490r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
'__inference_re_lu_7_layer_call_fn_17495e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
B__inference_re_lu_8_layer_call_and_return_conditional_losses_17582r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
'__inference_re_lu_8_layer_call_fn_17587e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
D__inference_reshape_4_layer_call_and_return_conditional_losses_17738\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? |
)__inference_reshape_4_layer_call_fn_17743O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_reshape_5_layer_call_and_return_conditional_losses_17398\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? |
)__inference_reshape_5_layer_call_fn_17403O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_sequential_4_layer_call_and_return_conditional_losses_14831|4GH5678IJ9:;<KL=>?@>?;
4?1
'?$
dense_8_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_14885|4H5G678J9I:;<L=K>?@>?;
4?1
'?$
dense_8_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_17096u4GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_17193u4H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_4_layer_call_fn_14983o4GH5678IJ9:;<KL=>?@>?;
4?1
'?$
dense_8_input?????????
p

 
? "???????????
,__inference_sequential_4_layer_call_fn_15080o4H5G678J9I:;<L=K>?@>?;
4?1
'?$
dense_8_input?????????
p 

 
? "???????????
,__inference_sequential_4_layer_call_fn_17236h4GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "???????????
,__inference_sequential_4_layer_call_fn_17279h4H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
G__inference_sequential_5_layer_call_and_return_conditional_losses_13769|(AB)*+,CD-./0EF123??<
5?2
(?%
dense_10_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_13821|(B)A*+,D-C./0F1E23??<
5?2
(?%
dense_10_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_16726t(AB)*+,CD-./0EF1237?4
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_16869t(B)A*+,D-C./0F1E237?4
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
,__inference_sequential_5_layer_call_fn_13915o(AB)*+,CD-./0EF123??<
5?2
(?%
dense_10_input?????????
p

 
? "???????????
,__inference_sequential_5_layer_call_fn_14008o(B)A*+,D-C./0F1E23??<
5?2
(?%
dense_10_input?????????
p 

 
? "???????????
,__inference_sequential_5_layer_call_fn_16910g(AB)*+,CD-./0EF1237?4
-?*
 ?
inputs?????????
p

 
? "???????????
,__inference_sequential_5_layer_call_fn_16951g(B)A*+,D-C./0F1E237?4
-?*
 ?
inputs?????????
p 

 
? "???????????
G__inference_sequential_6_layer_call_and_return_conditional_losses_15329?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@C?@
9?6
,?)
sequential_5_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_15409?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@C?@
9?6
,?)
sequential_5_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_16141?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_16377?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_6_layer_call_fn_15569?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@C?@
9?6
,?)
sequential_5_input?????????
p

 
? "???????????
,__inference_sequential_6_layer_call_fn_15728?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@C?@
9?6
,?)
sequential_5_input?????????
p 

 
? "???????????
,__inference_sequential_6_layer_call_fn_16456z%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "???????????
,__inference_sequential_6_layer_call_fn_16535z%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_signature_wrapper_15809?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@Q?N
? 
G?D
B
sequential_5_input,?)
sequential_5_input?????????";?8
6
sequential_4&?#
sequential_4?????????