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
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
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
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
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
regularization_losses
Nmetrics
trainable_variables
	variables
Onon_trainable_variables

Players
Qlayer_metrics
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
regularization_losses
?metrics
trainable_variables
	variables
?non_trainable_variables
?layers
?layer_metrics
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
$regularization_losses
?metrics
%trainable_variables
&	variables
?non_trainable_variables
?layers
?layer_metrics
TR
VARIABLE_VALUEdense_6/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_9/gamma0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_9/beta0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose_2/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv1d_transpose_2/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_10/gamma0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_10/beta0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose_3/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv1d_transpose_3/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_11/gamma0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_11/beta1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_4/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_6/gamma1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_6/beta1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d_2/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv1d_2/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_7/gamma1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_7/beta1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d_3/kernel1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv1d_3/bias1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_8/gamma1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_8/beta1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_5/kernel1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_5/bias1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_10/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_10/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/21/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/22/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/27/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/28/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/33/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/34/.ATTRIBUTES/VARIABLE_VALUE
 
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

0
1
 
 

(0

(0
?
 ?layer_regularization_losses
Rregularization_losses
?metrics
Strainable_variables
T	variables
?non_trainable_variables
?layers
?layer_metrics
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
Wregularization_losses
?metrics
Xtrainable_variables
Y	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
[regularization_losses
?metrics
\trainable_variables
]	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
_regularization_losses
?metrics
`trainable_variables
a	variables
?non_trainable_variables
?layers
?layer_metrics
 

+0
,1

+0
,1
?
 ?layer_regularization_losses
cregularization_losses
?metrics
dtrainable_variables
e	variables
?non_trainable_variables
?layers
?layer_metrics
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
hregularization_losses
?metrics
itrainable_variables
j	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
lregularization_losses
?metrics
mtrainable_variables
n	variables
?non_trainable_variables
?layers
?layer_metrics
 

/0
01

/0
01
?
 ?layer_regularization_losses
pregularization_losses
?metrics
qtrainable_variables
r	variables
?non_trainable_variables
?layers
?layer_metrics
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
uregularization_losses
?metrics
vtrainable_variables
w	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
yregularization_losses
?metrics
ztrainable_variables
{	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
}regularization_losses
?metrics
~trainable_variables
	variables
?non_trainable_variables
?layers
?layer_metrics
 

30

30
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
*
A0
B1
C2
D3
E4
F5
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
 

40

40
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 

70
81

70
81
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 

;0
<1

;0
<1
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 

?0
@1

?0
@1
?
 ?layer_regularization_losses
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
 
 
*
G0
H1
I2
J3
K4
L5
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
 
?
"serving_default_sequential_3_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_3_inputdense_6/kernel%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/betaconv1d_transpose_2/kernelconv1d_transpose_2/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betaconv1d_transpose_3/kernelconv1d_transpose_3/bias&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betadense_7/kerneldense_4/kernel%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/betaconv1d_2/kernelconv1d_2/bias%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/betaconv1d_3/kernelconv1d_3/bias%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/betadense_5/kerneldense_5/bias*1
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
GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_8073
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp-conv1d_transpose_2/kernel/Read/ReadVariableOp+conv1d_transpose_2/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp-conv1d_transpose_3/kernel/Read/ReadVariableOp+conv1d_transpose_3/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOpConst*2
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
__inference__traced_save_10567
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kernelbatch_normalization_9/gammabatch_normalization_9/betaconv1d_transpose_2/kernelconv1d_transpose_2/biasbatch_normalization_10/gammabatch_normalization_10/betaconv1d_transpose_3/kernelconv1d_transpose_3/biasbatch_normalization_11/gammabatch_normalization_11/betadense_7/kerneldense_4/kernelbatch_normalization_6/gammabatch_normalization_6/betaconv1d_2/kernelconv1d_2/biasbatch_normalization_7/gammabatch_normalization_7/betaconv1d_3/kernelconv1d_3/biasbatch_normalization_8/gammabatch_normalization_8/betadense_5/kerneldense_5/bias!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance"batch_normalization_11/moving_mean&batch_normalization_11/moving_variance!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*1
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
!__inference__traced_restore_10688??*
?
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_5931

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
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6401

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
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_10424

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
?
?
5__inference_batch_normalization_10_layer_call_fn_9736

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_55602
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
4__inference_batch_normalization_9_layer_call_fn_9639

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_54032
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
?0
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5750

inputs
assignmovingavg_5725
assignmovingavg_1_5731)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5725*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5725*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5725*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5725*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5725AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5725*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5731*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5731*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5731*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5731*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5731AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5731*
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
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10087

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
?:
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_7303

inputs
dense_4_7252
batch_normalization_6_7255
batch_normalization_6_7257
batch_normalization_6_7259
batch_normalization_6_7261
conv1d_2_7266
conv1d_2_7268
batch_normalization_7_7271
batch_normalization_7_7273
batch_normalization_7_7275
batch_normalization_7_7277
conv1d_3_7281
conv1d_3_7283
batch_normalization_8_7286
batch_normalization_8_7288
batch_normalization_8_7290
batch_normalization_8_7292
dense_5_7297
dense_5_7299
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_7252*
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
GPU 2J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_67032!
dense_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_6_7255batch_normalization_6_7257batch_normalization_6_7259batch_normalization_6_7261*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_64012/
-batch_normalization_6/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_67552
leaky_re_lu_3/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_67762
reshape_2/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_7266conv1d_2_7268*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_67992"
 conv1d_2/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_7_7271batch_normalization_7_7273batch_normalization_7_7275batch_normalization_7_7277*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68702/
-batch_normalization_7/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_69112
leaky_re_lu_4/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_3_7281conv1d_3_7283*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_69342"
 conv1d_3/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_8_7286batch_normalization_8_7288batch_normalization_8_7290batch_normalization_8_7292*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_70052/
-batch_normalization_8/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_70462
leaky_re_lu_5/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_70602
flatten_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_5_7297dense_5_7299*
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
GPU 2J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_70782!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_re_lu_4_layer_call_fn_9759

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
GPU 2J 8? *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_59312
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
?
c
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_6911

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
?
?
A__inference_dense_7_layer_call_and_return_conditional_losses_9876

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
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_10022

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
?
?
1__inference_conv1d_transpose_3_layer_call_fn_5654

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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_56442
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
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6870

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
5__inference_batch_normalization_7_layer_call_fn_10195

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68702
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
?
?
+__inference_sequential_4_layer_call_fn_8799

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
GPU 2J 8? *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_79152
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
?
I
-__inference_leaky_re_lu_4_layer_call_fn_10205

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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_69112
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
?
?
5__inference_batch_normalization_7_layer_call_fn_10100

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_65082
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
4__inference_batch_normalization_6_layer_call_fn_9979

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_64012
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
4__inference_batch_normalization_6_layer_call_fn_9966

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_63682
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
?
l
&__inference_dense_7_layer_call_fn_9883

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
GPU 2J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60202
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
c
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_6755

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
?
?
+__inference_sequential_4_layer_call_fn_7992
sequential_3_input
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
StatefulPartitionedCallStatefulPartitionedCallsequential_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_79152
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
_user_specified_namesequential_3_input
?
_
C__inference_reshape_3_layer_call_and_return_conditional_losses_5878

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
?
?
5__inference_batch_normalization_8_layer_call_fn_10380

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69852
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
?
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_10398

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
?
?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9815

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
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10347

inputs
assignmovingavg_10322
assignmovingavg_1_10328)
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
loc:@AssignMovingAvg/10322*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10322*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10322*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10322*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10322AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/10322*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10328*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10328*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10328*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10328*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10328AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10328*
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
?0
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6985

inputs
assignmovingavg_6960
assignmovingavg_1_6966)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6960*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6960*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6960*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6960*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6960AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6960*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6966*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6966*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6966*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6966*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6966AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6966*
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
?
?
+__inference_sequential_4_layer_call_fn_7833
sequential_3_input
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
StatefulPartitionedCallStatefulPartitionedCallsequential_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_77562
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
_user_specified_namesequential_3_input
?
E
)__inference_flatten_2_layer_call_fn_10414

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
GPU 2J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_70602
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
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_10200

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
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10285

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
??
?.
__inference__wrapped_model_5274
sequential_3_inputD
@sequential_4_sequential_3_dense_6_matmul_readvariableop_resourceU
Qsequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_resourceY
Usequential_4_sequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_2_resourcef
bsequential_4_sequential_3_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resourceP
Lsequential_4_sequential_3_conv1d_transpose_2_biasadd_readvariableop_resourceV
Rsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_resourceZ
Vsequential_4_sequential_3_batch_normalization_10_batchnorm_mul_readvariableop_resourceX
Tsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_1_resourceX
Tsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_2_resourcef
bsequential_4_sequential_3_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resourceP
Lsequential_4_sequential_3_conv1d_transpose_3_biasadd_readvariableop_resourceV
Rsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_resourceZ
Vsequential_4_sequential_3_batch_normalization_11_batchnorm_mul_readvariableop_resourceX
Tsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_1_resourceX
Tsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_2_resourceD
@sequential_4_sequential_3_dense_7_matmul_readvariableop_resourceD
@sequential_4_sequential_2_dense_4_matmul_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_2_resourceR
Nsequential_4_sequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resourceF
Bsequential_4_sequential_2_conv1d_2_biasadd_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_2_resourceR
Nsequential_4_sequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resourceF
Bsequential_4_sequential_2_conv1d_3_biasadd_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_2_resourceD
@sequential_4_sequential_2_dense_5_matmul_readvariableop_resourceE
Asequential_4_sequential_2_dense_5_biasadd_readvariableop_resource
identity??Hsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp?Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1?Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2?Lsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?Hsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp?Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1?Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2?Lsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?Hsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp?Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1?Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2?Lsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?9sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOp?Esequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?9sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOp?Esequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?7sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp?8sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp?7sequential_4/sequential_2/dense_5/MatMul/ReadVariableOp?Isequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2?Msequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?Isequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2?Msequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?Hsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1?Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2?Lsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?Csequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?Ysequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?Csequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?Ysequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?7sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp?7sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp?
7sequential_4/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp?
(sequential_4/sequential_3/dense_6/MatMulMatMulsequential_3_input?sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_3/dense_6/MatMul?
Hsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?
?sequential_4/sequential_3/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2A
?sequential_4/sequential_3/batch_normalization_9/batchnorm/add/y?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/addAddV2Psequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_3/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/add?
?sequential_4/sequential_3/batch_normalization_9/batchnorm/RsqrtRsqrtAsequential_4/sequential_3/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_3/batch_normalization_9/batchnorm/Rsqrt?
Lsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/mulMulCsequential_4/sequential_3/batch_normalization_9/batchnorm/Rsqrt:y:0Tsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/mul?
?sequential_4/sequential_3/batch_normalization_9/batchnorm/mul_1Mul2sequential_4/sequential_3/dense_6/MatMul:product:0Asequential_4/sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?sequential_4/sequential_3/batch_normalization_9/batchnorm/mul_1?
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1?
?sequential_4/sequential_3/batch_normalization_9/batchnorm/mul_2MulRsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_3/batch_normalization_9/batchnorm/mul_2?
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_3_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/subSubRsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_3/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_3/batch_normalization_9/batchnorm/sub?
?sequential_4/sequential_3/batch_normalization_9/batchnorm/add_1AddV2Csequential_4/sequential_3/batch_normalization_9/batchnorm/mul_1:z:0Asequential_4/sequential_3/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?sequential_4/sequential_3/batch_normalization_9/batchnorm/add_1?
&sequential_4/sequential_3/re_lu_3/ReluReluCsequential_4/sequential_3/batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_4/sequential_3/re_lu_3/Relu?
)sequential_4/sequential_3/reshape_3/ShapeShape4sequential_4/sequential_3/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2+
)sequential_4/sequential_3/reshape_3/Shape?
7sequential_4/sequential_3/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_4/sequential_3/reshape_3/strided_slice/stack?
9sequential_4/sequential_3/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/sequential_3/reshape_3/strided_slice/stack_1?
9sequential_4/sequential_3/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/sequential_3/reshape_3/strided_slice/stack_2?
1sequential_4/sequential_3/reshape_3/strided_sliceStridedSlice2sequential_4/sequential_3/reshape_3/Shape:output:0@sequential_4/sequential_3/reshape_3/strided_slice/stack:output:0Bsequential_4/sequential_3/reshape_3/strided_slice/stack_1:output:0Bsequential_4/sequential_3/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_4/sequential_3/reshape_3/strided_slice?
3sequential_4/sequential_3/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_4/sequential_3/reshape_3/Reshape/shape/1?
3sequential_4/sequential_3/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_4/sequential_3/reshape_3/Reshape/shape/2?
1sequential_4/sequential_3/reshape_3/Reshape/shapePack:sequential_4/sequential_3/reshape_3/strided_slice:output:0<sequential_4/sequential_3/reshape_3/Reshape/shape/1:output:0<sequential_4/sequential_3/reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:23
1sequential_4/sequential_3/reshape_3/Reshape/shape?
+sequential_4/sequential_3/reshape_3/ReshapeReshape4sequential_4/sequential_3/re_lu_3/Relu:activations:0:sequential_4/sequential_3/reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2-
+sequential_4/sequential_3/reshape_3/Reshape?
2sequential_4/sequential_3/conv1d_transpose_2/ShapeShape4sequential_4/sequential_3/reshape_3/Reshape:output:0*
T0*
_output_shapes
:24
2sequential_4/sequential_3/conv1d_transpose_2/Shape?
@sequential_4/sequential_3/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack?
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_1?
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_2?
:sequential_4/sequential_3/conv1d_transpose_2/strided_sliceStridedSlice;sequential_4/sequential_3/conv1d_transpose_2/Shape:output:0Isequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack:output:0Ksequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_1:output:0Ksequential_4/sequential_3/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_4/sequential_3/conv1d_transpose_2/strided_slice?
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack?
Dsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_1?
Dsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_2?
<sequential_4/sequential_3/conv1d_transpose_2/strided_slice_1StridedSlice;sequential_4/sequential_3/conv1d_transpose_2/Shape:output:0Ksequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack:output:0Msequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_1:output:0Msequential_4/sequential_3/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_4/sequential_3/conv1d_transpose_2/strided_slice_1?
2sequential_4/sequential_3/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_4/sequential_3/conv1d_transpose_2/mul/y?
0sequential_4/sequential_3/conv1d_transpose_2/mulMulEsequential_4/sequential_3/conv1d_transpose_2/strided_slice_1:output:0;sequential_4/sequential_3/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_4/sequential_3/conv1d_transpose_2/mul?
4sequential_4/sequential_3/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_4/sequential_3/conv1d_transpose_2/stack/2?
2sequential_4/sequential_3/conv1d_transpose_2/stackPackCsequential_4/sequential_3/conv1d_transpose_2/strided_slice:output:04sequential_4/sequential_3/conv1d_transpose_2/mul:z:0=sequential_4/sequential_3/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:24
2sequential_4/sequential_3/conv1d_transpose_2/stack?
Lsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
Hsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims4sequential_4/sequential_3/reshape_3/Reshape:output:0Usequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2J
Hsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims?
Ysequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpbsequential_4_sequential_3_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02[
Ysequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Nsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2P
Nsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
Jsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsasequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Wsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2L
Jsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
Qsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
Ksequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice;sequential_4/sequential_3/conv1d_transpose_2/stack:output:0Zsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0\sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0\sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2M
Ksequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice?
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
Usequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2W
Usequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
Usequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2W
Usequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
Msequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice;sequential_4/sequential_3/conv1d_transpose_2/stack:output:0\sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0^sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0^sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2O
Msequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1?
Msequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1?
Isequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Isequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/axis?
Dsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concatConcatV2Tsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Vsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Vsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0Rsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat?
=sequential_4/sequential_3/conv1d_transpose_2/conv1d_transposeConv2DBackpropInputMsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/concat:output:0Ssequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0Qsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2?
=sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose?
Esequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/SqueezeSqueezeFsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2G
Esequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/Squeeze?
Csequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpLsequential_4_sequential_3_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?
4sequential_4/sequential_3/conv1d_transpose_2/BiasAddBiasAddNsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0Ksequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????26
4sequential_4/sequential_3/conv1d_transpose_2/BiasAdd?
Isequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOpRsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?
@sequential_4/sequential_3/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_4/sequential_3/batch_normalization_10/batchnorm/add/y?
>sequential_4/sequential_3/batch_normalization_10/batchnorm/addAddV2Qsequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp:value:0Isequential_4/sequential_3/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_10/batchnorm/add?
@sequential_4/sequential_3/batch_normalization_10/batchnorm/RsqrtRsqrtBsequential_4/sequential_3/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_4/sequential_3/batch_normalization_10/batchnorm/Rsqrt?
Msequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_4_sequential_3_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?
>sequential_4/sequential_3/batch_normalization_10/batchnorm/mulMulDsequential_4/sequential_3/batch_normalization_10/batchnorm/Rsqrt:y:0Usequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_10/batchnorm/mul?
@sequential_4/sequential_3/batch_normalization_10/batchnorm/mul_1Mul=sequential_4/sequential_3/conv1d_transpose_2/BiasAdd:output:0Bsequential_4/sequential_3/batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_4/sequential_3/batch_normalization_10/batchnorm/mul_1?
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?
@sequential_4/sequential_3/batch_normalization_10/batchnorm/mul_2MulSsequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1:value:0Bsequential_4/sequential_3/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_4/sequential_3/batch_normalization_10/batchnorm/mul_2?
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_4_sequential_3_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2?
>sequential_4/sequential_3/batch_normalization_10/batchnorm/subSubSsequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2:value:0Dsequential_4/sequential_3/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_10/batchnorm/sub?
@sequential_4/sequential_3/batch_normalization_10/batchnorm/add_1AddV2Dsequential_4/sequential_3/batch_normalization_10/batchnorm/mul_1:z:0Bsequential_4/sequential_3/batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_4/sequential_3/batch_normalization_10/batchnorm/add_1?
&sequential_4/sequential_3/re_lu_4/ReluReluDsequential_4/sequential_3/batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2(
&sequential_4/sequential_3/re_lu_4/Relu?
2sequential_4/sequential_3/conv1d_transpose_3/ShapeShape4sequential_4/sequential_3/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:24
2sequential_4/sequential_3/conv1d_transpose_3/Shape?
@sequential_4/sequential_3/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack?
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_1?
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_2?
:sequential_4/sequential_3/conv1d_transpose_3/strided_sliceStridedSlice;sequential_4/sequential_3/conv1d_transpose_3/Shape:output:0Isequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack:output:0Ksequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_1:output:0Ksequential_4/sequential_3/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_4/sequential_3/conv1d_transpose_3/strided_slice?
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack?
Dsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_1?
Dsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_2?
<sequential_4/sequential_3/conv1d_transpose_3/strided_slice_1StridedSlice;sequential_4/sequential_3/conv1d_transpose_3/Shape:output:0Ksequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack:output:0Msequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_1:output:0Msequential_4/sequential_3/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_4/sequential_3/conv1d_transpose_3/strided_slice_1?
2sequential_4/sequential_3/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_4/sequential_3/conv1d_transpose_3/mul/y?
0sequential_4/sequential_3/conv1d_transpose_3/mulMulEsequential_4/sequential_3/conv1d_transpose_3/strided_slice_1:output:0;sequential_4/sequential_3/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_4/sequential_3/conv1d_transpose_3/mul?
4sequential_4/sequential_3/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_4/sequential_3/conv1d_transpose_3/stack/2?
2sequential_4/sequential_3/conv1d_transpose_3/stackPackCsequential_4/sequential_3/conv1d_transpose_3/strided_slice:output:04sequential_4/sequential_3/conv1d_transpose_3/mul:z:0=sequential_4/sequential_3/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:24
2sequential_4/sequential_3/conv1d_transpose_3/stack?
Lsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
Hsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims4sequential_4/sequential_3/re_lu_4/Relu:activations:0Usequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2J
Hsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims?
Ysequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpbsequential_4_sequential_3_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02[
Ysequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Nsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2P
Nsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
Jsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsasequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Wsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2L
Jsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
Qsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
Ksequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice;sequential_4/sequential_3/conv1d_transpose_3/stack:output:0Zsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0\sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0\sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2M
Ksequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice?
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
Usequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2W
Usequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
Usequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2W
Usequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
Msequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice;sequential_4/sequential_3/conv1d_transpose_3/stack:output:0\sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0^sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0^sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2O
Msequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1?
Msequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1?
Isequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Isequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/axis?
Dsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concatConcatV2Tsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Vsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Vsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0Rsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat?
=sequential_4/sequential_3/conv1d_transpose_3/conv1d_transposeConv2DBackpropInputMsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/concat:output:0Ssequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0Qsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2?
=sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose?
Esequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/SqueezeSqueezeFsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2G
Esequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/Squeeze?
Csequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpLsequential_4_sequential_3_conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?
4sequential_4/sequential_3/conv1d_transpose_3/BiasAddBiasAddNsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0Ksequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????26
4sequential_4/sequential_3/conv1d_transpose_3/BiasAdd?
Isequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOpRsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02K
Isequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?
@sequential_4/sequential_3/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2B
@sequential_4/sequential_3/batch_normalization_11/batchnorm/add/y?
>sequential_4/sequential_3/batch_normalization_11/batchnorm/addAddV2Qsequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp:value:0Isequential_4/sequential_3/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_11/batchnorm/add?
@sequential_4/sequential_3/batch_normalization_11/batchnorm/RsqrtRsqrtBsequential_4/sequential_3/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:2B
@sequential_4/sequential_3/batch_normalization_11/batchnorm/Rsqrt?
Msequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpVsequential_4_sequential_3_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02O
Msequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?
>sequential_4/sequential_3/batch_normalization_11/batchnorm/mulMulDsequential_4/sequential_3/batch_normalization_11/batchnorm/Rsqrt:y:0Usequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_11/batchnorm/mul?
@sequential_4/sequential_3/batch_normalization_11/batchnorm/mul_1Mul=sequential_4/sequential_3/conv1d_transpose_3/BiasAdd:output:0Bsequential_4/sequential_3/batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_4/sequential_3/batch_normalization_11/batchnorm/mul_1?
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpTsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02M
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?
@sequential_4/sequential_3/batch_normalization_11/batchnorm/mul_2MulSsequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1:value:0Bsequential_4/sequential_3/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:2B
@sequential_4/sequential_3/batch_normalization_11/batchnorm/mul_2?
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpTsequential_4_sequential_3_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02M
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2?
>sequential_4/sequential_3/batch_normalization_11/batchnorm/subSubSsequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2:value:0Dsequential_4/sequential_3/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2@
>sequential_4/sequential_3/batch_normalization_11/batchnorm/sub?
@sequential_4/sequential_3/batch_normalization_11/batchnorm/add_1AddV2Dsequential_4/sequential_3/batch_normalization_11/batchnorm/mul_1:z:0Bsequential_4/sequential_3/batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2B
@sequential_4/sequential_3/batch_normalization_11/batchnorm/add_1?
&sequential_4/sequential_3/re_lu_5/ReluReluDsequential_4/sequential_3/batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2(
&sequential_4/sequential_3/re_lu_5/Relu?
)sequential_4/sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)sequential_4/sequential_3/flatten_3/Const?
+sequential_4/sequential_3/flatten_3/ReshapeReshape4sequential_4/sequential_3/re_lu_5/Relu:activations:02sequential_4/sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_4/sequential_3/flatten_3/Reshape?
7sequential_4/sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp?
(sequential_4/sequential_3/dense_7/MatMulMatMul4sequential_4/sequential_3/flatten_3/Reshape:output:0?sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_3/dense_7/MatMul?
&sequential_4/sequential_3/dense_7/TanhTanh2sequential_4/sequential_3/dense_7/MatMul:product:0*
T0*'
_output_shapes
:?????????2(
&sequential_4/sequential_3/dense_7/Tanh?
7sequential_4/sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp?
(sequential_4/sequential_2/dense_4/MatMulMatMul*sequential_4/sequential_3/dense_7/Tanh:y:0?sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_2/dense_4/MatMul?
Hsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp?
?sequential_4/sequential_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2A
?sequential_4/sequential_2/batch_normalization_6/batchnorm/add/y?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/add?
?sequential_4/sequential_2/batch_normalization_6/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_6/batchnorm/Rsqrt?
Lsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_6/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/mul?
?sequential_4/sequential_2/batch_normalization_6/batchnorm/mul_1Mul2sequential_4/sequential_2/dense_4/MatMul:product:0Asequential_4/sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_6/batchnorm/mul_1?
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1?
?sequential_4/sequential_2/batch_normalization_6/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_6/batchnorm/mul_2?
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_6/batchnorm/sub?
?sequential_4/sequential_2/batch_normalization_6/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_6/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_6/batchnorm/add_1?
1sequential_4/sequential_2/leaky_re_lu_3/LeakyRelu	LeakyReluCsequential_4/sequential_2/batch_normalization_6/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>23
1sequential_4/sequential_2/leaky_re_lu_3/LeakyRelu?
)sequential_4/sequential_2/reshape_2/ShapeShape?sequential_4/sequential_2/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2+
)sequential_4/sequential_2/reshape_2/Shape?
7sequential_4/sequential_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_4/sequential_2/reshape_2/strided_slice/stack?
9sequential_4/sequential_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/sequential_2/reshape_2/strided_slice/stack_1?
9sequential_4/sequential_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/sequential_2/reshape_2/strided_slice/stack_2?
1sequential_4/sequential_2/reshape_2/strided_sliceStridedSlice2sequential_4/sequential_2/reshape_2/Shape:output:0@sequential_4/sequential_2/reshape_2/strided_slice/stack:output:0Bsequential_4/sequential_2/reshape_2/strided_slice/stack_1:output:0Bsequential_4/sequential_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_4/sequential_2/reshape_2/strided_slice?
3sequential_4/sequential_2/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_4/sequential_2/reshape_2/Reshape/shape/1?
3sequential_4/sequential_2/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_4/sequential_2/reshape_2/Reshape/shape/2?
1sequential_4/sequential_2/reshape_2/Reshape/shapePack:sequential_4/sequential_2/reshape_2/strided_slice:output:0<sequential_4/sequential_2/reshape_2/Reshape/shape/1:output:0<sequential_4/sequential_2/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:23
1sequential_4/sequential_2/reshape_2/Reshape/shape?
+sequential_4/sequential_2/reshape_2/ReshapeReshape?sequential_4/sequential_2/leaky_re_lu_3/LeakyRelu:activations:0:sequential_4/sequential_2/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2-
+sequential_4/sequential_2/reshape_2/Reshape?
8sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims/dim?
4sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims
ExpandDims4sequential_4/sequential_2/reshape_2/Reshape:output:0Asequential_4/sequential_2/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????26
4sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims?
Esequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpNsequential_4_sequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02G
Esequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
:sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/dim?
6sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1
ExpandDimsMsequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Csequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:28
6sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1?
)sequential_4/sequential_2/conv1d_2/conv1dConv2D=sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims:output:0?sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)sequential_4/sequential_2/conv1d_2/conv1d?
1sequential_4/sequential_2/conv1d_2/conv1d/SqueezeSqueeze2sequential_4/sequential_2/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????23
1sequential_4/sequential_2/conv1d_2/conv1d/Squeeze?
9sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOp?
*sequential_4/sequential_2/conv1d_2/BiasAddBiasAdd:sequential_4/sequential_2/conv1d_2/conv1d/Squeeze:output:0Asequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2,
*sequential_4/sequential_2/conv1d_2/BiasAdd?
Hsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp?
?sequential_4/sequential_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2A
?sequential_4/sequential_2/batch_normalization_7/batchnorm/add/y?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/add?
?sequential_4/sequential_2/batch_normalization_7/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_7/batchnorm/Rsqrt?
Lsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_7/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/mul?
?sequential_4/sequential_2/batch_normalization_7/batchnorm/mul_1Mul3sequential_4/sequential_2/conv1d_2/BiasAdd:output:0Asequential_4/sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_7/batchnorm/mul_1?
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1?
?sequential_4/sequential_2/batch_normalization_7/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_7/batchnorm/mul_2?
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_7/batchnorm/sub?
?sequential_4/sequential_2/batch_normalization_7/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_7/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_7/batchnorm/add_1?
1sequential_4/sequential_2/leaky_re_lu_4/LeakyRelu	LeakyReluCsequential_4/sequential_2/batch_normalization_7/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>23
1sequential_4/sequential_2/leaky_re_lu_4/LeakyRelu?
8sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims/dim?
4sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims
ExpandDims?sequential_4/sequential_2/leaky_re_lu_4/LeakyRelu:activations:0Asequential_4/sequential_2/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????26
4sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims?
Esequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpNsequential_4_sequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02G
Esequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
:sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/dim?
6sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1
ExpandDimsMsequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0Csequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:28
6sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1?
)sequential_4/sequential_2/conv1d_3/conv1dConv2D=sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims:output:0?sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)sequential_4/sequential_2/conv1d_3/conv1d?
1sequential_4/sequential_2/conv1d_3/conv1d/SqueezeSqueeze2sequential_4/sequential_2/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????23
1sequential_4/sequential_2/conv1d_3/conv1d/Squeeze?
9sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_2_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOp?
*sequential_4/sequential_2/conv1d_3/BiasAddBiasAdd:sequential_4/sequential_2/conv1d_3/conv1d/Squeeze:output:0Asequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2,
*sequential_4/sequential_2/conv1d_3/BiasAdd?
Hsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp?
?sequential_4/sequential_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2A
?sequential_4/sequential_2/batch_normalization_8/batchnorm/add/y?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/add?
?sequential_4/sequential_2/batch_normalization_8/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_8/batchnorm/Rsqrt?
Lsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_8/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/mul?
?sequential_4/sequential_2/batch_normalization_8/batchnorm/mul_1Mul3sequential_4/sequential_2/conv1d_3/BiasAdd:output:0Asequential_4/sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_8/batchnorm/mul_1?
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1?
?sequential_4/sequential_2/batch_normalization_8/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_8/batchnorm/mul_2?
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_8/batchnorm/sub?
?sequential_4/sequential_2/batch_normalization_8/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_8/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2A
?sequential_4/sequential_2/batch_normalization_8/batchnorm/add_1?
1sequential_4/sequential_2/leaky_re_lu_5/LeakyRelu	LeakyReluCsequential_4/sequential_2/batch_normalization_8/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>23
1sequential_4/sequential_2/leaky_re_lu_5/LeakyRelu?
)sequential_4/sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2+
)sequential_4/sequential_2/flatten_2/Const?
+sequential_4/sequential_2/flatten_2/ReshapeReshape?sequential_4/sequential_2/leaky_re_lu_5/LeakyRelu:activations:02sequential_4/sequential_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_4/sequential_2/flatten_2/Reshape?
7sequential_4/sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_2/dense_5/MatMul/ReadVariableOp?
(sequential_4/sequential_2/dense_5/MatMulMatMul4sequential_4/sequential_2/flatten_2/Reshape:output:0?sequential_4/sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sequential_4/sequential_2/dense_5/MatMul?
8sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp?
)sequential_4/sequential_2/dense_5/BiasAddBiasAdd2sequential_4/sequential_2/dense_5/MatMul:product:0@sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sequential_4/sequential_2/dense_5/BiasAdd?
IdentityIdentity2sequential_4/sequential_2/dense_5/BiasAdd:output:0I^sequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:^sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOpF^sequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:^sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOpF^sequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp8^sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp9^sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp8^sequential_4/sequential_2/dense_5/MatMul/ReadVariableOpJ^sequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOpL^sequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1L^sequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2N^sequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOpJ^sequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOpL^sequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1L^sequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2N^sequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOpK^sequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1K^sequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2M^sequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOpD^sequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpZ^sequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpD^sequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpZ^sequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp8^sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp8^sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2?
Hsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp2?
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_12?
Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_22?
Lsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2?
Hsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp2?
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_12?
Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_22?
Lsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2?
Hsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp2?
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_12?
Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_22?
Lsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2v
9sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOp9sequential_4/sequential_2/conv1d_2/BiasAdd/ReadVariableOp2?
Esequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpEsequential_4/sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2v
9sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOp9sequential_4/sequential_2/conv1d_3/BiasAdd/ReadVariableOp2?
Esequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpEsequential_4/sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2r
7sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_4/MatMul/ReadVariableOp2t
8sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp8sequential_4/sequential_2/dense_5/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_2/dense_5/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_5/MatMul/ReadVariableOp2?
Isequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOpIsequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp2?
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_12?
Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2Ksequential_4/sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_22?
Msequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOpMsequential_4/sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp2?
Isequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOpIsequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp2?
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_12?
Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2Ksequential_4/sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_22?
Msequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOpMsequential_4/sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp2?
Hsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOpHsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp2?
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_12?
Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2Jsequential_4/sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_22?
Lsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOpLsequential_4/sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp2?
Csequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpCsequential_4/sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp2?
Ysequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpYsequential_4/sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
Csequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpCsequential_4/sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp2?
Ysequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpYsequential_4/sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2r
7sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp7sequential_4/sequential_3/dense_6/MatMul/ReadVariableOp2r
7sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp7sequential_4/sequential_3/dense_7/MatMul/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_3_input
?
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_9846

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
?/
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5370

inputs
assignmovingavg_5345
assignmovingavg_1_5351)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5345*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5345*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5345*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5345*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5345AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5345*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5351*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5351*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5351*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5351*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5351AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5351*
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
E
)__inference_reshape_2_layer_call_fn_10007

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
GPU 2J 8? *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_67762
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
?<
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_6085
dense_6_input
dense_6_6036
batch_normalization_9_6039
batch_normalization_9_6041
batch_normalization_9_6043
batch_normalization_9_6045
conv1d_transpose_2_6050
conv1d_transpose_2_6052
batch_normalization_10_6055
batch_normalization_10_6057
batch_normalization_10_6059
batch_normalization_10_6061
conv1d_transpose_3_6065
conv1d_transpose_3_6067
batch_normalization_11_6070
batch_normalization_11_6072
batch_normalization_11_6074
batch_normalization_11_6076
dense_7_6081
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCalldense_6_inputdense_6_6036*
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
GPU 2J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_58052!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_6039batch_normalization_9_6041batch_normalization_9_6043batch_normalization_9_6045*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_54032/
-batch_normalization_9/StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_58572
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
GPU 2J 8? *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_58782
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_6050conv1d_transpose_2_6052*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_54542,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_6055batch_normalization_10_6057batch_normalization_10_6059batch_normalization_10_6061*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_559320
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
GPU 2J 8? *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_59312
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_6065conv1d_transpose_3_6067*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_56442,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_6070batch_normalization_11_6072batch_normalization_11_6074batch_normalization_11_6076*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_578320
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
GPU 2J 8? *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_59842
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
GPU 2J 8? *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_60042
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_6081*
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
GPU 2J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60202!
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
?0
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10265

inputs
assignmovingavg_10240
assignmovingavg_1_10246)
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
loc:@AssignMovingAvg/10240*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10240*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10240*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10240*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10240AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/10240*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10246*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10246*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10246*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10246*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10246AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10246*
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
1__inference_conv1d_transpose_2_layer_call_fn_5464

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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_54542
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
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5403

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
?N
?
__inference__traced_save_10567
file_prefix-
)savev2_dense_6_kernel_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop8
4savev2_conv1d_transpose_2_kernel_read_readvariableop6
2savev2_conv1d_transpose_2_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableop8
4savev2_conv1d_transpose_3_kernel_read_readvariableop6
2savev2_conv1d_transpose_3_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop4savev2_conv1d_transpose_2_kernel_read_readvariableop2savev2_conv1d_transpose_2_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop4savev2_conv1d_transpose_3_kernel_read_readvariableop2savev2_conv1d_transpose_3_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop)savev2_dense_4_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
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
?
H
,__inference_leaky_re_lu_3_layer_call_fn_9989

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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_67552
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
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10169

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
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9795

inputs
assignmovingavg_9770
assignmovingavg_1_9776)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9770*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9770*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9770*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9770*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9770AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9770*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9776*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9776*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9776*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9776*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9776AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9776*
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
?
?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_6799

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
?
?
F__inference_sequential_4_layer_call_and_return_conditional_losses_7915

inputs
sequential_3_7838
sequential_3_7840
sequential_3_7842
sequential_3_7844
sequential_3_7846
sequential_3_7848
sequential_3_7850
sequential_3_7852
sequential_3_7854
sequential_3_7856
sequential_3_7858
sequential_3_7860
sequential_3_7862
sequential_3_7864
sequential_3_7866
sequential_3_7868
sequential_3_7870
sequential_3_7872
sequential_2_7875
sequential_2_7877
sequential_2_7879
sequential_2_7881
sequential_2_7883
sequential_2_7885
sequential_2_7887
sequential_2_7889
sequential_2_7891
sequential_2_7893
sequential_2_7895
sequential_2_7897
sequential_2_7899
sequential_2_7901
sequential_2_7903
sequential_2_7905
sequential_2_7907
sequential_2_7909
sequential_2_7911
identity??$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_7838sequential_3_7840sequential_3_7842sequential_3_7844sequential_3_7846sequential_3_7848sequential_3_7850sequential_3_7852sequential_3_7854sequential_3_7856sequential_3_7858sequential_3_7860sequential_3_7862sequential_3_7864sequential_3_7866sequential_3_7868sequential_3_7870sequential_3_7872*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_62332&
$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_2_7875sequential_2_7877sequential_2_7879sequential_2_7881sequential_2_7883sequential_2_7885sequential_2_7887sequential_2_7889sequential_2_7891sequential_2_7893sequential_2_7895sequential_2_7897sequential_2_7899sequential_2_7901sequential_2_7903sequential_2_7905sequential_2_7907sequential_2_7909sequential_2_7911*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_73032&
$sequential_2/StatefulPartitionedCall?
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_2_layer_call_fn_7344
dense_4_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_73032
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
_user_specified_namedense_4_input
?:
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_7206

inputs
dense_4_7155
batch_normalization_6_7158
batch_normalization_6_7160
batch_normalization_6_7162
batch_normalization_6_7164
conv1d_2_7169
conv1d_2_7171
batch_normalization_7_7174
batch_normalization_7_7176
batch_normalization_7_7178
batch_normalization_7_7180
conv1d_3_7184
conv1d_3_7186
batch_normalization_8_7189
batch_normalization_8_7191
batch_normalization_8_7193
batch_normalization_8_7195
dense_5_7200
dense_5_7202
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_7155*
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
GPU 2J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_67032!
dense_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_6_7158batch_normalization_6_7160batch_normalization_6_7162batch_normalization_6_7164*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_63682/
-batch_normalization_6/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_67552
leaky_re_lu_3/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_67762
reshape_2/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_7169conv1d_2_7171*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_67992"
 conv1d_2/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_7_7174batch_normalization_7_7176batch_normalization_7_7178batch_normalization_7_7180*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68502/
-batch_normalization_7/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_69112
leaky_re_lu_4/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_3_7184conv1d_3_7186*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_69342"
 conv1d_3/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_8_7189batch_normalization_8_7191batch_normalization_8_7193batch_normalization_8_7195*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69852/
-batch_normalization_8/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_70462
leaky_re_lu_5/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_70602
flatten_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_5_7200dense_5_7202*
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
GPU 2J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_70782!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6850

inputs
assignmovingavg_6825
assignmovingavg_1_6831)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6825*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6825*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6825*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6825*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6825AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6825*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6831*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6831*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6831*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6831*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6831AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6831*
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
?
l
&__inference_dense_6_layer_call_fn_9557

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
GPU 2J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_58052
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
?0
?
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_5454

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
5__inference_batch_normalization_7_layer_call_fn_10113

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_65412
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
?
l
&__inference_dense_4_layer_call_fn_9897

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
GPU 2J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_67032
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
?0
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10149

inputs
assignmovingavg_10124
assignmovingavg_1_10130)
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
loc:@AssignMovingAvg/10124*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10124*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10124*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10124*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10124AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/10124*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10130*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10130*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10130*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10130*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10130AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10130*
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
?
?
F__inference_sequential_4_layer_call_and_return_conditional_losses_7756

inputs
sequential_3_7679
sequential_3_7681
sequential_3_7683
sequential_3_7685
sequential_3_7687
sequential_3_7689
sequential_3_7691
sequential_3_7693
sequential_3_7695
sequential_3_7697
sequential_3_7699
sequential_3_7701
sequential_3_7703
sequential_3_7705
sequential_3_7707
sequential_3_7709
sequential_3_7711
sequential_3_7713
sequential_2_7716
sequential_2_7718
sequential_2_7720
sequential_2_7722
sequential_2_7724
sequential_2_7726
sequential_2_7728
sequential_2_7730
sequential_2_7732
sequential_2_7734
sequential_2_7736
sequential_2_7738
sequential_2_7740
sequential_2_7742
sequential_2_7744
sequential_2_7746
sequential_2_7748
sequential_2_7750
sequential_2_7752
identity??$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallinputssequential_3_7679sequential_3_7681sequential_3_7683sequential_3_7685sequential_3_7687sequential_3_7689sequential_3_7691sequential_3_7693sequential_3_7695sequential_3_7697sequential_3_7699sequential_3_7701sequential_3_7703sequential_3_7705sequential_3_7707sequential_3_7709sequential_3_7711sequential_3_7713*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_61402&
$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_2_7716sequential_2_7718sequential_2_7720sequential_2_7722sequential_2_7724sequential_2_7726sequential_2_7728sequential_2_7730sequential_2_7732sequential_2_7734sequential_2_7736sequential_2_7738sequential_2_7740sequential_2_7742sequential_2_7744sequential_2_7746sequential_2_7748sequential_2_7750sequential_2_7752*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_72062&
$sequential_2/StatefulPartitionedCall?
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_10_layer_call_fn_9749

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_55932
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
?/
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9933

inputs
assignmovingavg_9908
assignmovingavg_1_9914)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9908*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9908*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9908*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9908*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9908AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9908*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9914*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9914*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9914*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9914*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9914AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9914*
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
A__inference_dense_7_layer_call_and_return_conditional_losses_6020

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
??
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_9133

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
?
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_9754

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
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10367

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
?
D
(__inference_reshape_3_layer_call_fn_9667

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
GPU 2J 8? *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_58782
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
?
?
5__inference_batch_normalization_11_layer_call_fn_9828

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_57502
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
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_9644

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
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10067

inputs
assignmovingavg_10042
assignmovingavg_1_10048)
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
loc:@AssignMovingAvg/10042*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10042*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10042*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/10042*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10042AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/10042*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10048*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10048*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10048*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/10048*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10048AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/10048*
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
c
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_9984

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
?
?
+__inference_sequential_3_layer_call_fn_6272
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_62332
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
?:
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_7095
dense_4_input
dense_4_6712
batch_normalization_6_6741
batch_normalization_6_6743
batch_normalization_6_6745
batch_normalization_6_6747
conv1d_2_6810
conv1d_2_6812
batch_normalization_7_6897
batch_normalization_7_6899
batch_normalization_7_6901
batch_normalization_7_6903
conv1d_3_6945
conv1d_3_6947
batch_normalization_8_7032
batch_normalization_8_7034
batch_normalization_8_7036
batch_normalization_8_7038
dense_5_7089
dense_5_7091
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_6712*
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
GPU 2J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_67032!
dense_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_6_6741batch_normalization_6_6743batch_normalization_6_6745batch_normalization_6_6747*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_63682/
-batch_normalization_6/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_67552
leaky_re_lu_3/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_67762
reshape_2/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_6810conv1d_2_6812*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_67992"
 conv1d_2/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_7_6897batch_normalization_7_6899batch_normalization_7_6901batch_normalization_7_6903*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68502/
-batch_normalization_7/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_69112
leaky_re_lu_4/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_3_6945conv1d_3_6947*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_69342"
 conv1d_3/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_8_7032batch_normalization_8_7034batch_normalization_8_7036batch_normalization_8_7038*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69852/
-batch_normalization_8/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_70462
leaky_re_lu_5/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_70602
flatten_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_5_7089dense_5_7091*
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
GPU 2J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_70782!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_4_input
?:
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_7149
dense_4_input
dense_4_7098
batch_normalization_6_7101
batch_normalization_6_7103
batch_normalization_6_7105
batch_normalization_6_7107
conv1d_2_7112
conv1d_2_7114
batch_normalization_7_7117
batch_normalization_7_7119
batch_normalization_7_7121
batch_normalization_7_7123
conv1d_3_7127
conv1d_3_7129
batch_normalization_8_7132
batch_normalization_8_7134
batch_normalization_8_7136
batch_normalization_8_7138
dense_5_7143
dense_5_7145
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_7098*
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
GPU 2J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_67032!
dense_4/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_6_7101batch_normalization_6_7103batch_normalization_6_7105batch_normalization_6_7107*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_64012/
-batch_normalization_6/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_67552
leaky_re_lu_3/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_67762
reshape_2/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_7112conv1d_2_7114*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_67992"
 conv1d_2/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_7_7117batch_normalization_7_7119batch_normalization_7_7121batch_normalization_7_7123*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68702/
-batch_normalization_7/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_69112
leaky_re_lu_4/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv1d_3_7127conv1d_3_7129*
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
GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_69342"
 conv1d_3/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_8_7132batch_normalization_8_7134batch_normalization_8_7136batch_normalization_8_7138*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_70052/
-batch_normalization_8/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_70462
leaky_re_lu_5/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_70602
flatten_2/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_5_7143dense_5_7145*
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
GPU 2J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_70782!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_4_input
?
?
+__inference_sequential_2_layer_call_fn_9543

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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_73032
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
?0
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5560

inputs
assignmovingavg_5535
assignmovingavg_1_5541)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5535*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5535*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5535*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5535*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5535AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5535*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5541*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5541*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5541*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5541*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5541AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5541*
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
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6368

inputs
assignmovingavg_6343
assignmovingavg_1_6349)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6343*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6343*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6343*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6343*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6343AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6343*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6349*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6349*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6349*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6349*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6349AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6349*
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
?
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9613

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
?
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_9863

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
?
B
&__inference_re_lu_3_layer_call_fn_9649

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
GPU 2J 8? *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_58572
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
_
C__inference_reshape_3_layer_call_and_return_conditional_losses_9662

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
B
&__inference_re_lu_5_layer_call_fn_9851

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
GPU 2J 8? *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_59842
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
?
I
-__inference_leaky_re_lu_5_layer_call_fn_10403

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
GPU 2J 8? *P
fKRI
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_70462
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
?
?
"__inference_signature_wrapper_8073
sequential_3_input
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
StatefulPartitionedCallStatefulPartitionedCallsequential_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *(
f#R!
__inference__wrapped_model_52742
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
_user_specified_namesequential_3_input
?
?
4__inference_batch_normalization_9_layer_call_fn_9626

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_53702
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
?
?
A__inference_dense_4_layer_call_and_return_conditional_losses_9890

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
??
?
!__inference__traced_restore_10688
file_prefix#
assignvariableop_dense_6_kernel2
.assignvariableop_1_batch_normalization_9_gamma1
-assignvariableop_2_batch_normalization_9_beta0
,assignvariableop_3_conv1d_transpose_2_kernel.
*assignvariableop_4_conv1d_transpose_2_bias3
/assignvariableop_5_batch_normalization_10_gamma2
.assignvariableop_6_batch_normalization_10_beta0
,assignvariableop_7_conv1d_transpose_3_kernel.
*assignvariableop_8_conv1d_transpose_3_bias3
/assignvariableop_9_batch_normalization_11_gamma3
/assignvariableop_10_batch_normalization_11_beta&
"assignvariableop_11_dense_7_kernel&
"assignvariableop_12_dense_4_kernel3
/assignvariableop_13_batch_normalization_6_gamma2
.assignvariableop_14_batch_normalization_6_beta'
#assignvariableop_15_conv1d_2_kernel%
!assignvariableop_16_conv1d_2_bias3
/assignvariableop_17_batch_normalization_7_gamma2
.assignvariableop_18_batch_normalization_7_beta'
#assignvariableop_19_conv1d_3_kernel%
!assignvariableop_20_conv1d_3_bias3
/assignvariableop_21_batch_normalization_8_gamma2
.assignvariableop_22_batch_normalization_8_beta&
"assignvariableop_23_dense_5_kernel$
 assignvariableop_24_dense_5_bias9
5assignvariableop_25_batch_normalization_9_moving_mean=
9assignvariableop_26_batch_normalization_9_moving_variance:
6assignvariableop_27_batch_normalization_10_moving_mean>
:assignvariableop_28_batch_normalization_10_moving_variance:
6assignvariableop_29_batch_normalization_11_moving_mean>
:assignvariableop_30_batch_normalization_11_moving_variance9
5assignvariableop_31_batch_normalization_6_moving_mean=
9assignvariableop_32_batch_normalization_6_moving_variance9
5assignvariableop_33_batch_normalization_7_moving_mean=
9assignvariableop_34_batch_normalization_7_moving_variance9
5assignvariableop_35_batch_normalization_8_moving_mean=
9assignvariableop_36_batch_normalization_8_moving_variance
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
AssignVariableOp_3AssignVariableOp,assignvariableop_3_conv1d_transpose_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv1d_transpose_2_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp/assignvariableop_5_batch_normalization_10_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_10_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv1d_transpose_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv1d_transpose_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_11_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_11_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_7_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_6_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_6_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv1d_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv1d_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_7_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_batch_normalization_7_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv1d_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv1d_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_8_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_8_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_5_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp5assignvariableop_25_batch_normalization_9_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp9assignvariableop_26_batch_normalization_9_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_batch_normalization_10_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp:assignvariableop_28_batch_normalization_10_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_batch_normalization_11_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp:assignvariableop_30_batch_normalization_11_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_batch_normalization_6_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp9assignvariableop_32_batch_normalization_6_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp5assignvariableop_33_batch_normalization_7_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_normalization_7_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_batch_normalization_8_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp9assignvariableop_36_batch_normalization_8_moving_varianceIdentity_36:output:0"/device:CPU:0*
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
?
?
+__inference_sequential_4_layer_call_fn_8720

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
GPU 2J 8? *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_77562
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
C__inference_conv1d_3_layer_call_and_return_conditional_losses_10220

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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5593

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
+__inference_sequential_3_layer_call_fn_6179
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_61402
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
?/
?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9593

inputs
assignmovingavg_9568
assignmovingavg_1_9574)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9568*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9568*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9568*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9568*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9568AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9568*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9574*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9574*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9574*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9574*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9574AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9574*
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
?
A__inference_dense_4_layer_call_and_return_conditional_losses_6703

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
?
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_5857

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
c
G__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_7046

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
?0
?
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_5644

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
?;
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_6233

inputs
dense_6_6184
batch_normalization_9_6187
batch_normalization_9_6189
batch_normalization_9_6191
batch_normalization_9_6193
conv1d_transpose_2_6198
conv1d_transpose_2_6200
batch_normalization_10_6203
batch_normalization_10_6205
batch_normalization_10_6207
batch_normalization_10_6209
conv1d_transpose_3_6213
conv1d_transpose_3_6215
batch_normalization_11_6218
batch_normalization_11_6220
batch_normalization_11_6222
batch_normalization_11_6224
dense_7_6229
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_6184*
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
GPU 2J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_58052!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_6187batch_normalization_9_6189batch_normalization_9_6191batch_normalization_9_6193*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_54032/
-batch_normalization_9/StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_58572
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
GPU 2J 8? *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_58782
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_6198conv1d_transpose_2_6200*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_54542,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_6203batch_normalization_10_6205batch_normalization_10_6207batch_normalization_10_6209*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_559320
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
GPU 2J 8? *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_59312
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_6213conv1d_transpose_3_6215*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_56442,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_6218batch_normalization_11_6220batch_normalization_11_6222batch_normalization_11_6224*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_578320
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
GPU 2J 8? *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_59842
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
GPU 2J 8? *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_60042
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_6229*
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
GPU 2J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60202!
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
?
}
(__inference_conv1d_3_layer_call_fn_10229

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
GPU 2J 8? *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_69342
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
?
|
'__inference_dense_5_layer_call_fn_10433

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
GPU 2J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_70782
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
?
?
A__inference_dense_6_layer_call_and_return_conditional_losses_9550

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
?
?
F__inference_sequential_4_layer_call_and_return_conditional_losses_7673
sequential_3_input
sequential_3_7596
sequential_3_7598
sequential_3_7600
sequential_3_7602
sequential_3_7604
sequential_3_7606
sequential_3_7608
sequential_3_7610
sequential_3_7612
sequential_3_7614
sequential_3_7616
sequential_3_7618
sequential_3_7620
sequential_3_7622
sequential_3_7624
sequential_3_7626
sequential_3_7628
sequential_3_7630
sequential_2_7633
sequential_2_7635
sequential_2_7637
sequential_2_7639
sequential_2_7641
sequential_2_7643
sequential_2_7645
sequential_2_7647
sequential_2_7649
sequential_2_7651
sequential_2_7653
sequential_2_7655
sequential_2_7657
sequential_2_7659
sequential_2_7661
sequential_2_7663
sequential_2_7665
sequential_2_7667
sequential_2_7669
identity??$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallsequential_3_inputsequential_3_7596sequential_3_7598sequential_3_7600sequential_3_7602sequential_3_7604sequential_3_7606sequential_3_7608sequential_3_7610sequential_3_7612sequential_3_7614sequential_3_7616sequential_3_7618sequential_3_7620sequential_3_7622sequential_3_7624sequential_3_7626sequential_3_7628sequential_3_7630*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_62332&
$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_2_7633sequential_2_7635sequential_2_7637sequential_2_7639sequential_2_7641sequential_2_7643sequential_2_7645sequential_2_7647sequential_2_7649sequential_2_7651sequential_2_7653sequential_2_7655sequential_2_7657sequential_2_7659sequential_2_7661sequential_2_7663sequential_2_7665sequential_2_7667sequential_2_7669*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_73032&
$sequential_2/StatefulPartitionedCall?
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_3_input
?
D
(__inference_flatten_3_layer_call_fn_9868

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
GPU 2J 8? *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_60042
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
?0
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9703

inputs
assignmovingavg_9678
assignmovingavg_1_9684)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9678*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9678*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9678*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9678*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9678AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9678*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9684*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9684*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9684*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9684*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9684AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9684*
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
??
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_8990

inputs*
&dense_6_matmul_readvariableop_resource.
*batch_normalization_9_assignmovingavg_88130
,batch_normalization_9_assignmovingavg_1_8819?
;batch_normalization_9_batchnorm_mul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resourceL
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_2_biasadd_readvariableop_resource/
+batch_normalization_10_assignmovingavg_88901
-batch_normalization_10_assignmovingavg_1_8896@
<batch_normalization_10_batchnorm_mul_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resourceL
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_3_biasadd_readvariableop_resource/
+batch_normalization_11_assignmovingavg_89581
-batch_normalization_11_assignmovingavg_1_8964@
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
+batch_normalization_9/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_9/AssignMovingAvg/8813*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_9/AssignMovingAvg/decay?
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_9_assignmovingavg_8813*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp?
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_9/AssignMovingAvg/8813*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub?
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_9/AssignMovingAvg/8813*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul?
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_9_assignmovingavg_8813-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_9/AssignMovingAvg/8813*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_9/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg_1/8819*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_9/AssignMovingAvg_1/decay?
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_9_assignmovingavg_1_8819*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg_1/8819*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub?
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg_1/8819*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul?
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_9_assignmovingavg_1_8819/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg_1/8819*
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
,batch_normalization_10/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_10/AssignMovingAvg/8890*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_10/AssignMovingAvg/decay?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_10_assignmovingavg_8890*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp?
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_10/AssignMovingAvg/8890*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/sub?
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_10/AssignMovingAvg/8890*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul?
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_10_assignmovingavg_8890.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_10/AssignMovingAvg/8890*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_10/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg_1/8896*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_10/AssignMovingAvg_1/decay?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_10_assignmovingavg_1_8896*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg_1/8896*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/sub?
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg_1/8896*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul?
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_10_assignmovingavg_1_88960batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg_1/8896*
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
,batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_11/AssignMovingAvg/8958*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_11/AssignMovingAvg/decay?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_11_assignmovingavg_8958*
_output_shapes
:*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp?
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_11/AssignMovingAvg/8958*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/sub?
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_11/AssignMovingAvg/8958*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/mul?
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_11_assignmovingavg_8958.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_11/AssignMovingAvg/8958*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg_1/8964*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_11/AssignMovingAvg_1/decay?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_11_assignmovingavg_1_8964*
_output_shapes
:*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg_1/8964*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/sub?
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg_1/8964*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/mul?
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_11_assignmovingavg_1_89640batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg_1/8964*
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
?
?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9723

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
?0
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6508

inputs
assignmovingavg_6483
assignmovingavg_1_6489)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6483*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6483*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6483*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6483*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6483AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6483*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6489*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6489*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6489*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6489*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6489AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6489*
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
??
?'
F__inference_sequential_4_layer_call_and_return_conditional_losses_8641

inputs7
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
3sequential_3_dense_7_matmul_readvariableop_resource7
3sequential_2_dense_4_matmul_readvariableop_resourceH
Dsequential_2_batch_normalization_6_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_6_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_6_batchnorm_readvariableop_2_resourceE
Asequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource9
5sequential_2_conv1d_2_biasadd_readvariableop_resourceH
Dsequential_2_batch_normalization_7_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_7_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_7_batchnorm_readvariableop_2_resourceE
Asequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource9
5sequential_2_conv1d_3_biasadd_readvariableop_resourceH
Dsequential_2_batch_normalization_8_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_8_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_8_batchnorm_readvariableop_2_resource7
3sequential_2_dense_5_matmul_readvariableop_resource8
4sequential_2_dense_5_biasadd_readvariableop_resource
identity??;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?,sequential_2/conv1d_2/BiasAdd/ReadVariableOp?8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?,sequential_2/conv1d_3/BiasAdd/ReadVariableOp?8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2?@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2?@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1?=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2??sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?*sequential_3/dense_6/MatMul/ReadVariableOp?*sequential_3/dense_7/MatMul/ReadVariableOp?
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp?
sequential_3/dense_6/MatMulMatMulinputs2sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
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
sequential_3/dense_7/Tanh?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOp?
sequential_2/dense_4/MatMulMatMulsequential_3/dense_7/Tanh:y:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_4/MatMul?
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_6/batchnorm/add/y?
0sequential_2/batch_normalization_6/batchnorm/addAddV2Csequential_2/batch_normalization_6/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/add?
2sequential_2/batch_normalization_6/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_6/batchnorm/Rsqrt?
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_6/batchnorm/mulMul6sequential_2/batch_normalization_6/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/mul?
2sequential_2/batch_normalization_6/batchnorm/mul_1Mul%sequential_2/dense_4/MatMul:product:04sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_6/batchnorm/mul_1?
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_6/batchnorm/mul_2MulEsequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_6/batchnorm/mul_2?
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_6/batchnorm/subSubEsequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/sub?
2sequential_2/batch_normalization_6/batchnorm/add_1AddV26sequential_2/batch_normalization_6/batchnorm/mul_1:z:04sequential_2/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_6/batchnorm/add_1?
$sequential_2/leaky_re_lu_3/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_6/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_3/LeakyRelu?
sequential_2/reshape_2/ShapeShape2sequential_2/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_2/reshape_2/Shape?
*sequential_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_2/reshape_2/strided_slice/stack?
,sequential_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_2/strided_slice/stack_1?
,sequential_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_2/strided_slice/stack_2?
$sequential_2/reshape_2/strided_sliceStridedSlice%sequential_2/reshape_2/Shape:output:03sequential_2/reshape_2/strided_slice/stack:output:05sequential_2/reshape_2/strided_slice/stack_1:output:05sequential_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_2/reshape_2/strided_slice?
&sequential_2/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_2/Reshape/shape/1?
&sequential_2/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_2/Reshape/shape/2?
$sequential_2/reshape_2/Reshape/shapePack-sequential_2/reshape_2/strided_slice:output:0/sequential_2/reshape_2/Reshape/shape/1:output:0/sequential_2/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/reshape_2/Reshape/shape?
sequential_2/reshape_2/ReshapeReshape2sequential_2/leaky_re_lu_3/LeakyRelu:activations:0-sequential_2/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_2/reshape_2/Reshape?
+sequential_2/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_2/conv1d_2/conv1d/ExpandDims/dim?
'sequential_2/conv1d_2/conv1d/ExpandDims
ExpandDims'sequential_2/reshape_2/Reshape:output:04sequential_2/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_2/conv1d_2/conv1d/ExpandDims?
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_2/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/conv1d_2/conv1d/ExpandDims_1/dim?
)sequential_2/conv1d_2/conv1d/ExpandDims_1
ExpandDims@sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_2/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_2/conv1d_2/conv1d/ExpandDims_1?
sequential_2/conv1d_2/conv1dConv2D0sequential_2/conv1d_2/conv1d/ExpandDims:output:02sequential_2/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_2/conv1d_2/conv1d?
$sequential_2/conv1d_2/conv1d/SqueezeSqueeze%sequential_2/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_2/conv1d_2/conv1d/Squeeze?
,sequential_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv1d_2/BiasAdd/ReadVariableOp?
sequential_2/conv1d_2/BiasAddBiasAdd-sequential_2/conv1d_2/conv1d/Squeeze:output:04sequential_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_2/conv1d_2/BiasAdd?
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_7/batchnorm/add/y?
0sequential_2/batch_normalization_7/batchnorm/addAddV2Csequential_2/batch_normalization_7/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/add?
2sequential_2/batch_normalization_7/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_7/batchnorm/Rsqrt?
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_7/batchnorm/mulMul6sequential_2/batch_normalization_7/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/mul?
2sequential_2/batch_normalization_7/batchnorm/mul_1Mul&sequential_2/conv1d_2/BiasAdd:output:04sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_7/batchnorm/mul_1?
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_7/batchnorm/mul_2MulEsequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_7/batchnorm/mul_2?
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_7/batchnorm/subSubEsequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/sub?
2sequential_2/batch_normalization_7/batchnorm/add_1AddV26sequential_2/batch_normalization_7/batchnorm/mul_1:z:04sequential_2/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_7/batchnorm/add_1?
$sequential_2/leaky_re_lu_4/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_7/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_4/LeakyRelu?
+sequential_2/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_2/conv1d_3/conv1d/ExpandDims/dim?
'sequential_2/conv1d_3/conv1d/ExpandDims
ExpandDims2sequential_2/leaky_re_lu_4/LeakyRelu:activations:04sequential_2/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_2/conv1d_3/conv1d/ExpandDims?
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_2/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/conv1d_3/conv1d/ExpandDims_1/dim?
)sequential_2/conv1d_3/conv1d/ExpandDims_1
ExpandDims@sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_2/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_2/conv1d_3/conv1d/ExpandDims_1?
sequential_2/conv1d_3/conv1dConv2D0sequential_2/conv1d_3/conv1d/ExpandDims:output:02sequential_2/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_2/conv1d_3/conv1d?
$sequential_2/conv1d_3/conv1d/SqueezeSqueeze%sequential_2/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_2/conv1d_3/conv1d/Squeeze?
,sequential_2/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv1d_3/BiasAdd/ReadVariableOp?
sequential_2/conv1d_3/BiasAddBiasAdd-sequential_2/conv1d_3/conv1d/Squeeze:output:04sequential_2/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_2/conv1d_3/BiasAdd?
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_8/batchnorm/add/y?
0sequential_2/batch_normalization_8/batchnorm/addAddV2Csequential_2/batch_normalization_8/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/add?
2sequential_2/batch_normalization_8/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_8/batchnorm/Rsqrt?
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_8/batchnorm/mulMul6sequential_2/batch_normalization_8/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/mul?
2sequential_2/batch_normalization_8/batchnorm/mul_1Mul&sequential_2/conv1d_3/BiasAdd:output:04sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_8/batchnorm/mul_1?
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_8/batchnorm/mul_2MulEsequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_8/batchnorm/mul_2?
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_8/batchnorm/subSubEsequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/sub?
2sequential_2/batch_normalization_8/batchnorm/add_1AddV26sequential_2/batch_normalization_8/batchnorm/mul_1:z:04sequential_2/batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_8/batchnorm/add_1?
$sequential_2/leaky_re_lu_5/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_8/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_5/LeakyRelu?
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_2/flatten_2/Const?
sequential_2/flatten_2/ReshapeReshape2sequential_2/leaky_re_lu_5/LeakyRelu:activations:0%sequential_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_2/flatten_2/Reshape?
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOp?
sequential_2/dense_5/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_5/MatMul?
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOp?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_5/BiasAdd?
IdentityIdentity%sequential_2/dense_5/BiasAdd:output:0<^sequential_2/batch_normalization_6/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_7/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_8/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp-^sequential_2/conv1d_2/BiasAdd/ReadVariableOp9^sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp-^sequential_2/conv1d_3/BiasAdd/ReadVariableOp9^sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp=^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp=^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp<^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2@^sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp7^sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp7^sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2z
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_6/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_7/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_8/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2\
,sequential_2/conv1d_2/BiasAdd/ReadVariableOp,sequential_2/conv1d_2/BiasAdd/ReadVariableOp2t
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2\
,sequential_2/conv1d_3/BiasAdd/ReadVariableOp,sequential_2/conv1d_3/BiasAdd/ReadVariableOp2t
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2|
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
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_3_layer_call_fn_9215

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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_62332
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
?
}
(__inference_conv1d_2_layer_call_fn_10031

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
GPU 2J 8? *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_67992
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
?

?
+__inference_sequential_3_layer_call_fn_9174

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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_61402
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
_
C__inference_reshape_2_layer_call_and_return_conditional_losses_6776

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
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5783

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
?
?
F__inference_sequential_4_layer_call_and_return_conditional_losses_7593
sequential_3_input
sequential_3_7430
sequential_3_7432
sequential_3_7434
sequential_3_7436
sequential_3_7438
sequential_3_7440
sequential_3_7442
sequential_3_7444
sequential_3_7446
sequential_3_7448
sequential_3_7450
sequential_3_7452
sequential_3_7454
sequential_3_7456
sequential_3_7458
sequential_3_7460
sequential_3_7462
sequential_3_7464
sequential_2_7553
sequential_2_7555
sequential_2_7557
sequential_2_7559
sequential_2_7561
sequential_2_7563
sequential_2_7565
sequential_2_7567
sequential_2_7569
sequential_2_7571
sequential_2_7573
sequential_2_7575
sequential_2_7577
sequential_2_7579
sequential_2_7581
sequential_2_7583
sequential_2_7585
sequential_2_7587
sequential_2_7589
identity??$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallsequential_3_inputsequential_3_7430sequential_3_7432sequential_3_7434sequential_3_7436sequential_3_7438sequential_3_7440sequential_3_7442sequential_3_7444sequential_3_7446sequential_3_7448sequential_3_7450sequential_3_7452sequential_3_7454sequential_3_7456sequential_3_7458sequential_3_7460sequential_3_7462sequential_3_7464*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_61402&
$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_2_7553sequential_2_7555sequential_2_7557sequential_2_7559sequential_2_7561sequential_2_7563sequential_2_7565sequential_2_7567sequential_2_7569sequential_2_7571sequential_2_7573sequential_2_7575sequential_2_7577sequential_2_7579sequential_2_7581sequential_2_7583sequential_2_7585sequential_2_7587sequential_2_7589*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_72062&
$sequential_2/StatefulPartitionedCall?
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namesequential_3_input
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_10409

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
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_10002

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
?
?
5__inference_batch_normalization_11_layer_call_fn_9841

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_57832
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
?
+__inference_sequential_2_layer_call_fn_7247
dense_4_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_72062
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
_user_specified_namedense_4_input
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9953

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
5__inference_batch_normalization_8_layer_call_fn_10298

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66482
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
??
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_9457

inputs*
&dense_4_matmul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource?
;batch_normalization_8_batchnorm_mul_readvariableop_resource=
9batch_normalization_8_batchnorm_readvariableop_1_resource=
9batch_normalization_8_batchnorm_readvariableop_2_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??.batch_normalization_6/batchnorm/ReadVariableOp?0batch_normalization_6/batchnorm/ReadVariableOp_1?0batch_normalization_6/batchnorm/ReadVariableOp_2?2batch_normalization_6/batchnorm/mul/ReadVariableOp?.batch_normalization_7/batchnorm/ReadVariableOp?0batch_normalization_7/batchnorm/ReadVariableOp_1?0batch_normalization_7/batchnorm/ReadVariableOp_2?2batch_normalization_7/batchnorm/mul/ReadVariableOp?.batch_normalization_8/batchnorm/ReadVariableOp?0batch_normalization_8/batchnorm/ReadVariableOp_1?0batch_normalization_8/batchnorm/ReadVariableOp_2?2batch_normalization_8/batchnorm/mul/ReadVariableOp?conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp?
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_6/batchnorm/add/y?
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add?
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/Rsqrt?
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp?
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul?
%batch_normalization_6/batchnorm/mul_1Muldense_4/MatMul:product:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_6/batchnorm/mul_1?
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1?
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2?
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2?
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/sub?
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_6/batchnorm/add_1?
leaky_re_lu_3/LeakyRelu	LeakyRelu)batch_normalization_6/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_3/LeakyReluw
reshape_2/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
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
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshape%leaky_re_lu_3/LeakyRelu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_2/Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsreshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_2/BiasAdd?
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp?
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_7/batchnorm/add/y?
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add?
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt?
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp?
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul?
%batch_normalization_7/batchnorm/mul_1Mulconv1d_2/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_7/batchnorm/mul_1?
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1?
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2?
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2?
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub?
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_7/batchnorm/add_1?
leaky_re_lu_4/LeakyRelu	LeakyRelu)batch_normalization_7/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims%leaky_re_lu_4/LeakyRelu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_3/BiasAdd?
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOp?
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_8/batchnorm/add/y?
#batch_normalization_8/batchnorm/addAddV26batch_normalization_8/batchnorm/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add?
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt?
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp?
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul?
%batch_normalization_8/batchnorm/mul_1Mulconv1d_3/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_8/batchnorm/mul_1?
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_1?
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2?
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_2?
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub?
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_8/batchnorm/add_1?
leaky_re_lu_5/LeakyRelu	LeakyRelu)batch_normalization_8/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_5/LeakyRelus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshape%leaky_re_lu_5/LeakyRelu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshape?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_2/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
IdentityIdentitydense_5/BiasAdd:output:0/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp1^batch_normalization_8/batchnorm/ReadVariableOp_11^batch_normalization_8/batchnorm/ReadVariableOp_23^batch_normalization_8/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2d
0batch_normalization_8/batchnorm/ReadVariableOp_10batch_normalization_8/batchnorm/ReadVariableOp_12d
0batch_normalization_8/batchnorm/ReadVariableOp_20batch_normalization_8/batchnorm/ReadVariableOp_22h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_6004

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
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7005

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
5__inference_batch_normalization_7_layer_call_fn_10182

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_68502
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
?0
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6648

inputs
assignmovingavg_6623
assignmovingavg_1_6629)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6623*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6623*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6623*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/6623*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6623AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/6623*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6629*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6629*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6629*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/6629*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6629AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/6629*
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
??
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_9360

inputs*
&dense_4_matmul_readvariableop_resource.
*batch_normalization_6_assignmovingavg_92290
,batch_normalization_6_assignmovingavg_1_9235?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource.
*batch_normalization_7_assignmovingavg_92820
,batch_normalization_7_assignmovingavg_1_9288?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource.
*batch_normalization_8_assignmovingavg_93260
,batch_normalization_8_assignmovingavg_1_9332?
;batch_normalization_8_batchnorm_mul_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_6/AssignMovingAvg/ReadVariableOp?;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_6/batchnorm/ReadVariableOp?2batch_normalization_6/batchnorm/mul/ReadVariableOp?9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_7/AssignMovingAvg/ReadVariableOp?;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_7/batchnorm/ReadVariableOp?2batch_normalization_7/batchnorm/mul/ReadVariableOp?9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_8/AssignMovingAvg/ReadVariableOp?;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_8/batchnorm/ReadVariableOp?2batch_normalization_8/batchnorm/mul/ReadVariableOp?conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices?
"batch_normalization_6/moments/meanMeandense_4/MatMul:product:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_6/moments/mean?
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_6/moments/StopGradient?
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_4/MatMul:product:03batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????21
/batch_normalization_6/moments/SquaredDifference?
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices?
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_6/moments/variance?
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze?
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1?
+batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/9229*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_6/AssignMovingAvg/decay?
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_6_assignmovingavg_9229*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp?
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/9229*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/sub?
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/9229*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mul?
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_6_assignmovingavg_9229-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/9229*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/9235*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_6/AssignMovingAvg_1/decay?
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_1_9235*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/9235*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/sub?
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/9235*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mul?
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_1_9235/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/9235*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_6/batchnorm/add/y?
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add?
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/Rsqrt?
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp?
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul?
%batch_normalization_6/batchnorm/mul_1Muldense_4/MatMul:product:0'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_6/batchnorm/mul_1?
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2?
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp?
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/sub?
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_6/batchnorm/add_1?
leaky_re_lu_3/LeakyRelu	LeakyRelu)batch_normalization_6/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_3/LeakyReluw
reshape_2/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
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
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshape%leaky_re_lu_3/LeakyRelu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_2/Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsreshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_2/BiasAdd?
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indices?
"batch_normalization_7/moments/meanMeanconv1d_2/BiasAdd:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2$
"batch_normalization_7/moments/mean?
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
:2,
*batch_normalization_7/moments/StopGradient?
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferenceconv1d_2/BiasAdd:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????21
/batch_normalization_7/moments/SquaredDifference?
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_7/moments/variance/reduction_indices?
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2(
&batch_normalization_7/moments/variance?
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze?
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1?
+batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/9282*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_7/AssignMovingAvg/decay?
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_7_assignmovingavg_9282*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp?
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/9282*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/sub?
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/9282*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mul?
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_7_assignmovingavg_9282-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/9282*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/9288*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_7/AssignMovingAvg_1/decay?
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_1_9288*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/9288*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/sub?
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/9288*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mul?
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_1_9288/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/9288*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_7/batchnorm/add/y?
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add?
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt?
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp?
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul?
%batch_normalization_7/batchnorm/mul_1Mulconv1d_2/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_7/batchnorm/mul_1?
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2?
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp?
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub?
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_7/batchnorm/add_1?
leaky_re_lu_4/LeakyRelu	LeakyRelu)batch_normalization_7/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims%leaky_re_lu_4/LeakyRelu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_3/BiasAdd?
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_8/moments/mean/reduction_indices?
"batch_normalization_8/moments/meanMeanconv1d_3/BiasAdd:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2$
"batch_normalization_8/moments/mean?
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*"
_output_shapes
:2,
*batch_normalization_8/moments/StopGradient?
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferenceconv1d_3/BiasAdd:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????21
/batch_normalization_8/moments/SquaredDifference?
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_8/moments/variance/reduction_indices?
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2(
&batch_normalization_8/moments/variance?
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze?
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1?
+batch_normalization_8/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/9326*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_8/AssignMovingAvg/decay?
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_8_assignmovingavg_9326*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp?
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/9326*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/sub?
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/9326*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul?
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_8_assignmovingavg_9326-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/9326*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_8/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/9332*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_8/AssignMovingAvg_1/decay?
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_8_assignmovingavg_1_9332*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/9332*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/sub?
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/9332*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul?
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_8_assignmovingavg_1_9332/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/9332*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_8/batchnorm/add/y?
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add?
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt?
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp?
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul?
%batch_normalization_8/batchnorm/mul_1Mulconv1d_3/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_8/batchnorm/mul_1?
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2?
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOp?
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub?
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2'
%batch_normalization_8/batchnorm/add_1?
leaky_re_lu_5/LeakyRelu	LeakyRelu)batch_normalization_8/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_5/LeakyRelus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshape%leaky_re_lu_5/LeakyRelu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_2/Reshape?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_2/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?

IdentityIdentitydense_5/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp3^batch_normalization_8/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:?????????:::::::::::::::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_6033
dense_6_input
dense_6_5814
batch_normalization_9_5843
batch_normalization_9_5845
batch_normalization_9_5847
batch_normalization_9_5849
conv1d_transpose_2_5886
conv1d_transpose_2_5888
batch_normalization_10_5917
batch_normalization_10_5919
batch_normalization_10_5921
batch_normalization_10_5923
conv1d_transpose_3_5939
conv1d_transpose_3_5941
batch_normalization_11_5970
batch_normalization_11_5972
batch_normalization_11_5974
batch_normalization_11_5976
dense_7_6029
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCalldense_6_inputdense_6_5814*
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
GPU 2J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_58052!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_5843batch_normalization_9_5845batch_normalization_9_5847batch_normalization_9_5849*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_53702/
-batch_normalization_9/StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_58572
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
GPU 2J 8? *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_58782
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_5886conv1d_transpose_2_5888*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_54542,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_5917batch_normalization_10_5919batch_normalization_10_5921batch_normalization_10_5923*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_556020
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
GPU 2J 8? *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_59312
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_5939conv1d_transpose_3_5941*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_56442,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_5970batch_normalization_11_5972batch_normalization_11_5974batch_normalization_11_5976*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_575020
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
GPU 2J 8? *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_59842
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
GPU 2J 8? *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_60042
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_6029*
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
GPU 2J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60202!
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
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6681

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
A__inference_dense_6_layer_call_and_return_conditional_losses_5805

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
?
?
B__inference_conv1d_3_layer_call_and_return_conditional_losses_6934

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
?
+__inference_sequential_2_layer_call_fn_9500

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
GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_72062
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
?;
?
F__inference_sequential_3_layer_call_and_return_conditional_losses_6140

inputs
dense_6_6091
batch_normalization_9_6094
batch_normalization_9_6096
batch_normalization_9_6098
batch_normalization_9_6100
conv1d_transpose_2_6105
conv1d_transpose_2_6107
batch_normalization_10_6110
batch_normalization_10_6112
batch_normalization_10_6114
batch_normalization_10_6116
conv1d_transpose_3_6120
conv1d_transpose_3_6122
batch_normalization_11_6125
batch_normalization_11_6127
batch_normalization_11_6129
batch_normalization_11_6131
dense_7_6136
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_6091*
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
GPU 2J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_58052!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_6094batch_normalization_9_6096batch_normalization_9_6098batch_normalization_9_6100*
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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_53702/
-batch_normalization_9/StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_58572
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
GPU 2J 8? *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_58782
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_6105conv1d_transpose_2_6107*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_54542,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_6110batch_normalization_10_6112batch_normalization_10_6114batch_normalization_10_6116*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_556020
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
GPU 2J 8? *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_59312
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_6120conv1d_transpose_3_6122*
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
GPU 2J 8? *U
fPRN
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_56442,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_6125batch_normalization_11_6127batch_normalization_11_6129batch_normalization_11_6131*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_575020
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
GPU 2J 8? *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_59842
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
GPU 2J 8? *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_60042
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_6136*
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
GPU 2J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_60202!
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
?	
?
A__inference_dense_5_layer_call_and_return_conditional_losses_7078

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
?
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_7060

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
?
?
5__inference_batch_normalization_8_layer_call_fn_10311

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66812
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
?
?
5__inference_batch_normalization_8_layer_call_fn_10393

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
GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_70052
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
??
?-
F__inference_sequential_4_layer_call_and_return_conditional_losses_8405

inputs7
3sequential_3_dense_6_matmul_readvariableop_resource;
7sequential_3_batch_normalization_9_assignmovingavg_8087=
9sequential_3_batch_normalization_9_assignmovingavg_1_8093L
Hsequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resourceH
Dsequential_3_batch_normalization_9_batchnorm_readvariableop_resourceY
Usequential_3_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_3_conv1d_transpose_2_biasadd_readvariableop_resource<
8sequential_3_batch_normalization_10_assignmovingavg_8164>
:sequential_3_batch_normalization_10_assignmovingavg_1_8170M
Isequential_3_batch_normalization_10_batchnorm_mul_readvariableop_resourceI
Esequential_3_batch_normalization_10_batchnorm_readvariableop_resourceY
Usequential_3_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_3_conv1d_transpose_3_biasadd_readvariableop_resource<
8sequential_3_batch_normalization_11_assignmovingavg_8232>
:sequential_3_batch_normalization_11_assignmovingavg_1_8238M
Isequential_3_batch_normalization_11_batchnorm_mul_readvariableop_resourceI
Esequential_3_batch_normalization_11_batchnorm_readvariableop_resource7
3sequential_3_dense_7_matmul_readvariableop_resource7
3sequential_2_dense_4_matmul_readvariableop_resource;
7sequential_2_batch_normalization_6_assignmovingavg_8274=
9sequential_2_batch_normalization_6_assignmovingavg_1_8280L
Hsequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_6_batchnorm_readvariableop_resourceE
Asequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource9
5sequential_2_conv1d_2_biasadd_readvariableop_resource;
7sequential_2_batch_normalization_7_assignmovingavg_8327=
9sequential_2_batch_normalization_7_assignmovingavg_1_8333L
Hsequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_7_batchnorm_readvariableop_resourceE
Asequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource9
5sequential_2_conv1d_3_biasadd_readvariableop_resource;
7sequential_2_batch_normalization_8_assignmovingavg_8371=
9sequential_2_batch_normalization_8_assignmovingavg_1_8377L
Hsequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_8_batchnorm_readvariableop_resource7
3sequential_2_dense_5_matmul_readvariableop_resource8
4sequential_2_dense_5_biasadd_readvariableop_resource
identity??Fsequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOp?Asequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOp?Hsequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp?Csequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp?;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp??sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?Fsequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOp?Asequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOp?Hsequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp?Csequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp?;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp??sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?Fsequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOp?Asequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOp?Hsequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp?Csequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp??sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?,sequential_2/conv1d_2/BiasAdd/ReadVariableOp?8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?,sequential_2/conv1d_3/BiasAdd/ReadVariableOp?8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?Gsequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?Bsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOp?Isequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?Gsequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?Bsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOp?Isequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?Dsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?Fsequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?Asequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOp?Hsequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp?Csequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp??sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?*sequential_3/dense_6/MatMul/ReadVariableOp?*sequential_3/dense_7/MatMul/ReadVariableOp?
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp?
sequential_3/dense_6/MatMulMatMulinputs2sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_6/MatMul?
Asequential_3/batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_3/batch_normalization_9/moments/mean/reduction_indices?
/sequential_3/batch_normalization_9/moments/meanMean%sequential_3/dense_6/MatMul:product:0Jsequential_3/batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(21
/sequential_3/batch_normalization_9/moments/mean?
7sequential_3/batch_normalization_9/moments/StopGradientStopGradient8sequential_3/batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:29
7sequential_3/batch_normalization_9/moments/StopGradient?
<sequential_3/batch_normalization_9/moments/SquaredDifferenceSquaredDifference%sequential_3/dense_6/MatMul:product:0@sequential_3/batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2>
<sequential_3/batch_normalization_9/moments/SquaredDifference?
Esequential_3/batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_3/batch_normalization_9/moments/variance/reduction_indices?
3sequential_3/batch_normalization_9/moments/varianceMean@sequential_3/batch_normalization_9/moments/SquaredDifference:z:0Nsequential_3/batch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(25
3sequential_3/batch_normalization_9/moments/variance?
2sequential_3/batch_normalization_9/moments/SqueezeSqueeze8sequential_3/batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 24
2sequential_3/batch_normalization_9/moments/Squeeze?
4sequential_3/batch_normalization_9/moments/Squeeze_1Squeeze<sequential_3/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_3/batch_normalization_9/moments/Squeeze_1?
8sequential_3/batch_normalization_9/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_3/batch_normalization_9/AssignMovingAvg/8087*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential_3/batch_normalization_9/AssignMovingAvg/decay?
Asequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp7sequential_3_batch_normalization_9_assignmovingavg_8087*
_output_shapes
:*
dtype02C
Asequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOp?
6sequential_3/batch_normalization_9/AssignMovingAvg/subSubIsequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0;sequential_3/batch_normalization_9/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_3/batch_normalization_9/AssignMovingAvg/8087*
_output_shapes
:28
6sequential_3/batch_normalization_9/AssignMovingAvg/sub?
6sequential_3/batch_normalization_9/AssignMovingAvg/mulMul:sequential_3/batch_normalization_9/AssignMovingAvg/sub:z:0Asequential_3/batch_normalization_9/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_3/batch_normalization_9/AssignMovingAvg/8087*
_output_shapes
:28
6sequential_3/batch_normalization_9/AssignMovingAvg/mul?
Fsequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp7sequential_3_batch_normalization_9_assignmovingavg_8087:sequential_3/batch_normalization_9/AssignMovingAvg/mul:z:0B^sequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_3/batch_normalization_9/AssignMovingAvg/8087*
_output_shapes
 *
dtype02H
Fsequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?
:sequential_3/batch_normalization_9/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_3/batch_normalization_9/AssignMovingAvg_1/8093*
_output_shapes
: *
dtype0*
valueB
 *
?#<2<
:sequential_3/batch_normalization_9/AssignMovingAvg_1/decay?
Csequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp9sequential_3_batch_normalization_9_assignmovingavg_1_8093*
_output_shapes
:*
dtype02E
Csequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?
8sequential_3/batch_normalization_9/AssignMovingAvg_1/subSubKsequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_3/batch_normalization_9/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_3/batch_normalization_9/AssignMovingAvg_1/8093*
_output_shapes
:2:
8sequential_3/batch_normalization_9/AssignMovingAvg_1/sub?
8sequential_3/batch_normalization_9/AssignMovingAvg_1/mulMul<sequential_3/batch_normalization_9/AssignMovingAvg_1/sub:z:0Csequential_3/batch_normalization_9/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_3/batch_normalization_9/AssignMovingAvg_1/8093*
_output_shapes
:2:
8sequential_3/batch_normalization_9/AssignMovingAvg_1/mul?
Hsequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp9sequential_3_batch_normalization_9_assignmovingavg_1_8093<sequential_3/batch_normalization_9/AssignMovingAvg_1/mul:z:0D^sequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_3/batch_normalization_9/AssignMovingAvg_1/8093*
_output_shapes
 *
dtype02J
Hsequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp?
2sequential_3/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_3/batch_normalization_9/batchnorm/add/y?
0sequential_3/batch_normalization_9/batchnorm/addAddV2=sequential_3/batch_normalization_9/moments/Squeeze_1:output:0;sequential_3/batch_normalization_9/batchnorm/add/y:output:0*
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
2sequential_3/batch_normalization_9/batchnorm/mul_2Mul;sequential_3/batch_normalization_9/moments/Squeeze:output:04sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_3/batch_normalization_9/batchnorm/mul_2?
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOpDsequential_3_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?
0sequential_3/batch_normalization_9/batchnorm/subSubCsequential_3/batch_normalization_9/batchnorm/ReadVariableOp:value:06sequential_3/batch_normalization_9/batchnorm/mul_2:z:0*
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
Bsequential_3/batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_3/batch_normalization_10/moments/mean/reduction_indices?
0sequential_3/batch_normalization_10/moments/meanMean0sequential_3/conv1d_transpose_2/BiasAdd:output:0Ksequential_3/batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_3/batch_normalization_10/moments/mean?
8sequential_3/batch_normalization_10/moments/StopGradientStopGradient9sequential_3/batch_normalization_10/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_3/batch_normalization_10/moments/StopGradient?
=sequential_3/batch_normalization_10/moments/SquaredDifferenceSquaredDifference0sequential_3/conv1d_transpose_2/BiasAdd:output:0Asequential_3/batch_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_3/batch_normalization_10/moments/SquaredDifference?
Fsequential_3/batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_3/batch_normalization_10/moments/variance/reduction_indices?
4sequential_3/batch_normalization_10/moments/varianceMeanAsequential_3/batch_normalization_10/moments/SquaredDifference:z:0Osequential_3/batch_normalization_10/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_3/batch_normalization_10/moments/variance?
3sequential_3/batch_normalization_10/moments/SqueezeSqueeze9sequential_3/batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_3/batch_normalization_10/moments/Squeeze?
5sequential_3/batch_normalization_10/moments/Squeeze_1Squeeze=sequential_3/batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_3/batch_normalization_10/moments/Squeeze_1?
9sequential_3/batch_normalization_10/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_3/batch_normalization_10/AssignMovingAvg/8164*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_3/batch_normalization_10/AssignMovingAvg/decay?
Bsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_3_batch_normalization_10_assignmovingavg_8164*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOp?
7sequential_3/batch_normalization_10/AssignMovingAvg/subSubJsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0<sequential_3/batch_normalization_10/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_3/batch_normalization_10/AssignMovingAvg/8164*
_output_shapes
:29
7sequential_3/batch_normalization_10/AssignMovingAvg/sub?
7sequential_3/batch_normalization_10/AssignMovingAvg/mulMul;sequential_3/batch_normalization_10/AssignMovingAvg/sub:z:0Bsequential_3/batch_normalization_10/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_3/batch_normalization_10/AssignMovingAvg/8164*
_output_shapes
:29
7sequential_3/batch_normalization_10/AssignMovingAvg/mul?
Gsequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_3_batch_normalization_10_assignmovingavg_8164;sequential_3/batch_normalization_10/AssignMovingAvg/mul:z:0C^sequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_3/batch_normalization_10/AssignMovingAvg/8164*
_output_shapes
 *
dtype02I
Gsequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?
;sequential_3/batch_normalization_10/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_3/batch_normalization_10/AssignMovingAvg_1/8170*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_3/batch_normalization_10/AssignMovingAvg_1/decay?
Dsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_10_assignmovingavg_1_8170*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
9sequential_3/batch_normalization_10/AssignMovingAvg_1/subSubLsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_3/batch_normalization_10/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_3/batch_normalization_10/AssignMovingAvg_1/8170*
_output_shapes
:2;
9sequential_3/batch_normalization_10/AssignMovingAvg_1/sub?
9sequential_3/batch_normalization_10/AssignMovingAvg_1/mulMul=sequential_3/batch_normalization_10/AssignMovingAvg_1/sub:z:0Dsequential_3/batch_normalization_10/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_3/batch_normalization_10/AssignMovingAvg_1/8170*
_output_shapes
:2;
9sequential_3/batch_normalization_10/AssignMovingAvg_1/mul?
Isequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_3_batch_normalization_10_assignmovingavg_1_8170=sequential_3/batch_normalization_10/AssignMovingAvg_1/mul:z:0E^sequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_3/batch_normalization_10/AssignMovingAvg_1/8170*
_output_shapes
 *
dtype02K
Isequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_3/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_3/batch_normalization_10/batchnorm/add/y?
1sequential_3/batch_normalization_10/batchnorm/addAddV2>sequential_3/batch_normalization_10/moments/Squeeze_1:output:0<sequential_3/batch_normalization_10/batchnorm/add/y:output:0*
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
3sequential_3/batch_normalization_10/batchnorm/mul_2Mul<sequential_3/batch_normalization_10/moments/Squeeze:output:05sequential_3/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_10/batchnorm/mul_2?
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?
1sequential_3/batch_normalization_10/batchnorm/subSubDsequential_3/batch_normalization_10/batchnorm/ReadVariableOp:value:07sequential_3/batch_normalization_10/batchnorm/mul_2:z:0*
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
Bsequential_3/batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bsequential_3/batch_normalization_11/moments/mean/reduction_indices?
0sequential_3/batch_normalization_11/moments/meanMean0sequential_3/conv1d_transpose_3/BiasAdd:output:0Ksequential_3/batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(22
0sequential_3/batch_normalization_11/moments/mean?
8sequential_3/batch_normalization_11/moments/StopGradientStopGradient9sequential_3/batch_normalization_11/moments/mean:output:0*
T0*"
_output_shapes
:2:
8sequential_3/batch_normalization_11/moments/StopGradient?
=sequential_3/batch_normalization_11/moments/SquaredDifferenceSquaredDifference0sequential_3/conv1d_transpose_3/BiasAdd:output:0Asequential_3/batch_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2?
=sequential_3/batch_normalization_11/moments/SquaredDifference?
Fsequential_3/batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fsequential_3/batch_normalization_11/moments/variance/reduction_indices?
4sequential_3/batch_normalization_11/moments/varianceMeanAsequential_3/batch_normalization_11/moments/SquaredDifference:z:0Osequential_3/batch_normalization_11/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(26
4sequential_3/batch_normalization_11/moments/variance?
3sequential_3/batch_normalization_11/moments/SqueezeSqueeze9sequential_3/batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 25
3sequential_3/batch_normalization_11/moments/Squeeze?
5sequential_3/batch_normalization_11/moments/Squeeze_1Squeeze=sequential_3/batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 27
5sequential_3/batch_normalization_11/moments/Squeeze_1?
9sequential_3/batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_3/batch_normalization_11/AssignMovingAvg/8232*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9sequential_3/batch_normalization_11/AssignMovingAvg/decay?
Bsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_3_batch_normalization_11_assignmovingavg_8232*
_output_shapes
:*
dtype02D
Bsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOp?
7sequential_3/batch_normalization_11/AssignMovingAvg/subSubJsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0<sequential_3/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_3/batch_normalization_11/AssignMovingAvg/8232*
_output_shapes
:29
7sequential_3/batch_normalization_11/AssignMovingAvg/sub?
7sequential_3/batch_normalization_11/AssignMovingAvg/mulMul;sequential_3/batch_normalization_11/AssignMovingAvg/sub:z:0Bsequential_3/batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_3/batch_normalization_11/AssignMovingAvg/8232*
_output_shapes
:29
7sequential_3/batch_normalization_11/AssignMovingAvg/mul?
Gsequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_3_batch_normalization_11_assignmovingavg_8232;sequential_3/batch_normalization_11/AssignMovingAvg/mul:z:0C^sequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_3/batch_normalization_11/AssignMovingAvg/8232*
_output_shapes
 *
dtype02I
Gsequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?
;sequential_3/batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_3/batch_normalization_11/AssignMovingAvg_1/8238*
_output_shapes
: *
dtype0*
valueB
 *
?#<2=
;sequential_3/batch_normalization_11/AssignMovingAvg_1/decay?
Dsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_11_assignmovingavg_1_8238*
_output_shapes
:*
dtype02F
Dsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
9sequential_3/batch_normalization_11/AssignMovingAvg_1/subSubLsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:0>sequential_3/batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_3/batch_normalization_11/AssignMovingAvg_1/8238*
_output_shapes
:2;
9sequential_3/batch_normalization_11/AssignMovingAvg_1/sub?
9sequential_3/batch_normalization_11/AssignMovingAvg_1/mulMul=sequential_3/batch_normalization_11/AssignMovingAvg_1/sub:z:0Dsequential_3/batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_3/batch_normalization_11/AssignMovingAvg_1/8238*
_output_shapes
:2;
9sequential_3/batch_normalization_11/AssignMovingAvg_1/mul?
Isequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_3_batch_normalization_11_assignmovingavg_1_8238=sequential_3/batch_normalization_11/AssignMovingAvg_1/mul:z:0E^sequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_3/batch_normalization_11/AssignMovingAvg_1/8238*
_output_shapes
 *
dtype02K
Isequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?
3sequential_3/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_3/batch_normalization_11/batchnorm/add/y?
1sequential_3/batch_normalization_11/batchnorm/addAddV2>sequential_3/batch_normalization_11/moments/Squeeze_1:output:0<sequential_3/batch_normalization_11/batchnorm/add/y:output:0*
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
3sequential_3/batch_normalization_11/batchnorm/mul_2Mul<sequential_3/batch_normalization_11/moments/Squeeze:output:05sequential_3/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_11/batchnorm/mul_2?
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?
1sequential_3/batch_normalization_11/batchnorm/subSubDsequential_3/batch_normalization_11/batchnorm/ReadVariableOp:value:07sequential_3/batch_normalization_11/batchnorm/mul_2:z:0*
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
sequential_3/dense_7/Tanh?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOp?
sequential_2/dense_4/MatMulMatMulsequential_3/dense_7/Tanh:y:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_4/MatMul?
Asequential_2/batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/batch_normalization_6/moments/mean/reduction_indices?
/sequential_2/batch_normalization_6/moments/meanMean%sequential_2/dense_4/MatMul:product:0Jsequential_2/batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(21
/sequential_2/batch_normalization_6/moments/mean?
7sequential_2/batch_normalization_6/moments/StopGradientStopGradient8sequential_2/batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:29
7sequential_2/batch_normalization_6/moments/StopGradient?
<sequential_2/batch_normalization_6/moments/SquaredDifferenceSquaredDifference%sequential_2/dense_4/MatMul:product:0@sequential_2/batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2>
<sequential_2/batch_normalization_6/moments/SquaredDifference?
Esequential_2/batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_2/batch_normalization_6/moments/variance/reduction_indices?
3sequential_2/batch_normalization_6/moments/varianceMean@sequential_2/batch_normalization_6/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(25
3sequential_2/batch_normalization_6/moments/variance?
2sequential_2/batch_normalization_6/moments/SqueezeSqueeze8sequential_2/batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 24
2sequential_2/batch_normalization_6/moments/Squeeze?
4sequential_2/batch_normalization_6/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_2/batch_normalization_6/moments/Squeeze_1?
8sequential_2/batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_6/AssignMovingAvg/8274*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential_2/batch_normalization_6/AssignMovingAvg/decay?
Asequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp7sequential_2_batch_normalization_6_assignmovingavg_8274*
_output_shapes
:*
dtype02C
Asequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOp?
6sequential_2/batch_normalization_6/AssignMovingAvg/subSubIsequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_6/AssignMovingAvg/8274*
_output_shapes
:28
6sequential_2/batch_normalization_6/AssignMovingAvg/sub?
6sequential_2/batch_normalization_6/AssignMovingAvg/mulMul:sequential_2/batch_normalization_6/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_6/AssignMovingAvg/8274*
_output_shapes
:28
6sequential_2/batch_normalization_6/AssignMovingAvg/mul?
Fsequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp7sequential_2_batch_normalization_6_assignmovingavg_8274:sequential_2/batch_normalization_6/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_6/AssignMovingAvg/8274*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOp?
:sequential_2/batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_6/AssignMovingAvg_1/8280*
_output_shapes
: *
dtype0*
valueB
 *
?#<2<
:sequential_2/batch_normalization_6/AssignMovingAvg_1/decay?
Csequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp9sequential_2_batch_normalization_6_assignmovingavg_1_8280*
_output_shapes
:*
dtype02E
Csequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp?
8sequential_2/batch_normalization_6/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_6/AssignMovingAvg_1/8280*
_output_shapes
:2:
8sequential_2/batch_normalization_6/AssignMovingAvg_1/sub?
8sequential_2/batch_normalization_6/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_6/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_6/AssignMovingAvg_1/8280*
_output_shapes
:2:
8sequential_2/batch_normalization_6/AssignMovingAvg_1/mul?
Hsequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp9sequential_2_batch_normalization_6_assignmovingavg_1_8280<sequential_2/batch_normalization_6/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_6/AssignMovingAvg_1/8280*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp?
2sequential_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_6/batchnorm/add/y?
0sequential_2/batch_normalization_6/batchnorm/addAddV2=sequential_2/batch_normalization_6/moments/Squeeze_1:output:0;sequential_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/add?
2sequential_2/batch_normalization_6/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_6/batchnorm/Rsqrt?
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_6/batchnorm/mulMul6sequential_2/batch_normalization_6/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/mul?
2sequential_2/batch_normalization_6/batchnorm/mul_1Mul%sequential_2/dense_4/MatMul:product:04sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_6/batchnorm/mul_1?
2sequential_2/batch_normalization_6/batchnorm/mul_2Mul;sequential_2/batch_normalization_6/moments/Squeeze:output:04sequential_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_6/batchnorm/mul_2?
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp?
0sequential_2/batch_normalization_6/batchnorm/subSubCsequential_2/batch_normalization_6/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_6/batchnorm/sub?
2sequential_2/batch_normalization_6/batchnorm/add_1AddV26sequential_2/batch_normalization_6/batchnorm/mul_1:z:04sequential_2/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_6/batchnorm/add_1?
$sequential_2/leaky_re_lu_3/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_6/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_3/LeakyRelu?
sequential_2/reshape_2/ShapeShape2sequential_2/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_2/reshape_2/Shape?
*sequential_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_2/reshape_2/strided_slice/stack?
,sequential_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_2/strided_slice/stack_1?
,sequential_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/reshape_2/strided_slice/stack_2?
$sequential_2/reshape_2/strided_sliceStridedSlice%sequential_2/reshape_2/Shape:output:03sequential_2/reshape_2/strided_slice/stack:output:05sequential_2/reshape_2/strided_slice/stack_1:output:05sequential_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_2/reshape_2/strided_slice?
&sequential_2/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_2/Reshape/shape/1?
&sequential_2/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/reshape_2/Reshape/shape/2?
$sequential_2/reshape_2/Reshape/shapePack-sequential_2/reshape_2/strided_slice:output:0/sequential_2/reshape_2/Reshape/shape/1:output:0/sequential_2/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/reshape_2/Reshape/shape?
sequential_2/reshape_2/ReshapeReshape2sequential_2/leaky_re_lu_3/LeakyRelu:activations:0-sequential_2/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_2/reshape_2/Reshape?
+sequential_2/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_2/conv1d_2/conv1d/ExpandDims/dim?
'sequential_2/conv1d_2/conv1d/ExpandDims
ExpandDims'sequential_2/reshape_2/Reshape:output:04sequential_2/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_2/conv1d_2/conv1d/ExpandDims?
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_2/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/conv1d_2/conv1d/ExpandDims_1/dim?
)sequential_2/conv1d_2/conv1d/ExpandDims_1
ExpandDims@sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_2/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_2/conv1d_2/conv1d/ExpandDims_1?
sequential_2/conv1d_2/conv1dConv2D0sequential_2/conv1d_2/conv1d/ExpandDims:output:02sequential_2/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_2/conv1d_2/conv1d?
$sequential_2/conv1d_2/conv1d/SqueezeSqueeze%sequential_2/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_2/conv1d_2/conv1d/Squeeze?
,sequential_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv1d_2/BiasAdd/ReadVariableOp?
sequential_2/conv1d_2/BiasAddBiasAdd-sequential_2/conv1d_2/conv1d/Squeeze:output:04sequential_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_2/conv1d_2/BiasAdd?
Asequential_2/batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2C
Asequential_2/batch_normalization_7/moments/mean/reduction_indices?
/sequential_2/batch_normalization_7/moments/meanMean&sequential_2/conv1d_2/BiasAdd:output:0Jsequential_2/batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(21
/sequential_2/batch_normalization_7/moments/mean?
7sequential_2/batch_normalization_7/moments/StopGradientStopGradient8sequential_2/batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
:29
7sequential_2/batch_normalization_7/moments/StopGradient?
<sequential_2/batch_normalization_7/moments/SquaredDifferenceSquaredDifference&sequential_2/conv1d_2/BiasAdd:output:0@sequential_2/batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2>
<sequential_2/batch_normalization_7/moments/SquaredDifference?
Esequential_2/batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2G
Esequential_2/batch_normalization_7/moments/variance/reduction_indices?
3sequential_2/batch_normalization_7/moments/varianceMean@sequential_2/batch_normalization_7/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_7/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(25
3sequential_2/batch_normalization_7/moments/variance?
2sequential_2/batch_normalization_7/moments/SqueezeSqueeze8sequential_2/batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 24
2sequential_2/batch_normalization_7/moments/Squeeze?
4sequential_2/batch_normalization_7/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_2/batch_normalization_7/moments/Squeeze_1?
8sequential_2/batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_7/AssignMovingAvg/8327*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential_2/batch_normalization_7/AssignMovingAvg/decay?
Asequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp7sequential_2_batch_normalization_7_assignmovingavg_8327*
_output_shapes
:*
dtype02C
Asequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOp?
6sequential_2/batch_normalization_7/AssignMovingAvg/subSubIsequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_7/AssignMovingAvg/8327*
_output_shapes
:28
6sequential_2/batch_normalization_7/AssignMovingAvg/sub?
6sequential_2/batch_normalization_7/AssignMovingAvg/mulMul:sequential_2/batch_normalization_7/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_7/AssignMovingAvg/8327*
_output_shapes
:28
6sequential_2/batch_normalization_7/AssignMovingAvg/mul?
Fsequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp7sequential_2_batch_normalization_7_assignmovingavg_8327:sequential_2/batch_normalization_7/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_7/AssignMovingAvg/8327*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOp?
:sequential_2/batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_7/AssignMovingAvg_1/8333*
_output_shapes
: *
dtype0*
valueB
 *
?#<2<
:sequential_2/batch_normalization_7/AssignMovingAvg_1/decay?
Csequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp9sequential_2_batch_normalization_7_assignmovingavg_1_8333*
_output_shapes
:*
dtype02E
Csequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp?
8sequential_2/batch_normalization_7/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_7/AssignMovingAvg_1/8333*
_output_shapes
:2:
8sequential_2/batch_normalization_7/AssignMovingAvg_1/sub?
8sequential_2/batch_normalization_7/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_7/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_7/AssignMovingAvg_1/8333*
_output_shapes
:2:
8sequential_2/batch_normalization_7/AssignMovingAvg_1/mul?
Hsequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp9sequential_2_batch_normalization_7_assignmovingavg_1_8333<sequential_2/batch_normalization_7/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_7/AssignMovingAvg_1/8333*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp?
2sequential_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_7/batchnorm/add/y?
0sequential_2/batch_normalization_7/batchnorm/addAddV2=sequential_2/batch_normalization_7/moments/Squeeze_1:output:0;sequential_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/add?
2sequential_2/batch_normalization_7/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_7/batchnorm/Rsqrt?
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_7/batchnorm/mulMul6sequential_2/batch_normalization_7/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/mul?
2sequential_2/batch_normalization_7/batchnorm/mul_1Mul&sequential_2/conv1d_2/BiasAdd:output:04sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_7/batchnorm/mul_1?
2sequential_2/batch_normalization_7/batchnorm/mul_2Mul;sequential_2/batch_normalization_7/moments/Squeeze:output:04sequential_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_7/batchnorm/mul_2?
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp?
0sequential_2/batch_normalization_7/batchnorm/subSubCsequential_2/batch_normalization_7/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_7/batchnorm/sub?
2sequential_2/batch_normalization_7/batchnorm/add_1AddV26sequential_2/batch_normalization_7/batchnorm/mul_1:z:04sequential_2/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_7/batchnorm/add_1?
$sequential_2/leaky_re_lu_4/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_7/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_4/LeakyRelu?
+sequential_2/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential_2/conv1d_3/conv1d/ExpandDims/dim?
'sequential_2/conv1d_3/conv1d/ExpandDims
ExpandDims2sequential_2/leaky_re_lu_4/LeakyRelu:activations:04sequential_2/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_2/conv1d_3/conv1d/ExpandDims?
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
-sequential_2/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/conv1d_3/conv1d/ExpandDims_1/dim?
)sequential_2/conv1d_3/conv1d/ExpandDims_1
ExpandDims@sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_2/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential_2/conv1d_3/conv1d/ExpandDims_1?
sequential_2/conv1d_3/conv1dConv2D0sequential_2/conv1d_3/conv1d/ExpandDims:output:02sequential_2/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_2/conv1d_3/conv1d?
$sequential_2/conv1d_3/conv1d/SqueezeSqueeze%sequential_2/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2&
$sequential_2/conv1d_3/conv1d/Squeeze?
,sequential_2/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv1d_3/BiasAdd/ReadVariableOp?
sequential_2/conv1d_3/BiasAddBiasAdd-sequential_2/conv1d_3/conv1d/Squeeze:output:04sequential_2/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_2/conv1d_3/BiasAdd?
Asequential_2/batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2C
Asequential_2/batch_normalization_8/moments/mean/reduction_indices?
/sequential_2/batch_normalization_8/moments/meanMean&sequential_2/conv1d_3/BiasAdd:output:0Jsequential_2/batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(21
/sequential_2/batch_normalization_8/moments/mean?
7sequential_2/batch_normalization_8/moments/StopGradientStopGradient8sequential_2/batch_normalization_8/moments/mean:output:0*
T0*"
_output_shapes
:29
7sequential_2/batch_normalization_8/moments/StopGradient?
<sequential_2/batch_normalization_8/moments/SquaredDifferenceSquaredDifference&sequential_2/conv1d_3/BiasAdd:output:0@sequential_2/batch_normalization_8/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2>
<sequential_2/batch_normalization_8/moments/SquaredDifference?
Esequential_2/batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2G
Esequential_2/batch_normalization_8/moments/variance/reduction_indices?
3sequential_2/batch_normalization_8/moments/varianceMean@sequential_2/batch_normalization_8/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_8/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(25
3sequential_2/batch_normalization_8/moments/variance?
2sequential_2/batch_normalization_8/moments/SqueezeSqueeze8sequential_2/batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 24
2sequential_2/batch_normalization_8/moments/Squeeze?
4sequential_2/batch_normalization_8/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_2/batch_normalization_8/moments/Squeeze_1?
8sequential_2/batch_normalization_8/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_8/AssignMovingAvg/8371*
_output_shapes
: *
dtype0*
valueB
 *
?#<2:
8sequential_2/batch_normalization_8/AssignMovingAvg/decay?
Asequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp7sequential_2_batch_normalization_8_assignmovingavg_8371*
_output_shapes
:*
dtype02C
Asequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOp?
6sequential_2/batch_normalization_8/AssignMovingAvg/subSubIsequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_8/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_8/AssignMovingAvg/8371*
_output_shapes
:28
6sequential_2/batch_normalization_8/AssignMovingAvg/sub?
6sequential_2/batch_normalization_8/AssignMovingAvg/mulMul:sequential_2/batch_normalization_8/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_8/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@sequential_2/batch_normalization_8/AssignMovingAvg/8371*
_output_shapes
:28
6sequential_2/batch_normalization_8/AssignMovingAvg/mul?
Fsequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp7sequential_2_batch_normalization_8_assignmovingavg_8371:sequential_2/batch_normalization_8/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@sequential_2/batch_normalization_8/AssignMovingAvg/8371*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOp?
:sequential_2/batch_normalization_8/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_8/AssignMovingAvg_1/8377*
_output_shapes
: *
dtype0*
valueB
 *
?#<2<
:sequential_2/batch_normalization_8/AssignMovingAvg_1/decay?
Csequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp9sequential_2_batch_normalization_8_assignmovingavg_1_8377*
_output_shapes
:*
dtype02E
Csequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?
8sequential_2/batch_normalization_8/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_8/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_8/AssignMovingAvg_1/8377*
_output_shapes
:2:
8sequential_2/batch_normalization_8/AssignMovingAvg_1/sub?
8sequential_2/batch_normalization_8/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_8/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_8/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@sequential_2/batch_normalization_8/AssignMovingAvg_1/8377*
_output_shapes
:2:
8sequential_2/batch_normalization_8/AssignMovingAvg_1/mul?
Hsequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp9sequential_2_batch_normalization_8_assignmovingavg_1_8377<sequential_2/batch_normalization_8/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@sequential_2/batch_normalization_8/AssignMovingAvg_1/8377*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp?
2sequential_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_8/batchnorm/add/y?
0sequential_2/batch_normalization_8/batchnorm/addAddV2=sequential_2/batch_normalization_8/moments/Squeeze_1:output:0;sequential_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/add?
2sequential_2/batch_normalization_8/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_8/batchnorm/Rsqrt?
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_8/batchnorm/mulMul6sequential_2/batch_normalization_8/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/mul?
2sequential_2/batch_normalization_8/batchnorm/mul_1Mul&sequential_2/conv1d_3/BiasAdd:output:04sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_8/batchnorm/mul_1?
2sequential_2/batch_normalization_8/batchnorm/mul_2Mul;sequential_2/batch_normalization_8/moments/Squeeze:output:04sequential_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_8/batchnorm/mul_2?
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp?
0sequential_2/batch_normalization_8/batchnorm/subSubCsequential_2/batch_normalization_8/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_8/batchnorm/sub?
2sequential_2/batch_normalization_8/batchnorm/add_1AddV26sequential_2/batch_normalization_8/batchnorm/mul_1:z:04sequential_2/batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????24
2sequential_2/batch_normalization_8/batchnorm/add_1?
$sequential_2/leaky_re_lu_5/LeakyRelu	LeakyRelu6sequential_2/batch_normalization_8/batchnorm/add_1:z:0*+
_output_shapes
:?????????*
alpha%???>2&
$sequential_2/leaky_re_lu_5/LeakyRelu?
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_2/flatten_2/Const?
sequential_2/flatten_2/ReshapeReshape2sequential_2/leaky_re_lu_5/LeakyRelu:activations:0%sequential_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_2/flatten_2/Reshape?
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOp?
sequential_2/dense_5/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_5/MatMul?
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOp?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_5/BiasAdd?
IdentityIdentity%sequential_2/dense_5/BiasAdd:output:0G^sequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_6/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_7/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_8/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp-^sequential_2/conv1d_2/BiasAdd/ReadVariableOp9^sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp-^sequential_2/conv1d_3/BiasAdd/ReadVariableOp9^sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOpH^sequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOpC^sequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOpJ^sequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpE^sequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp=^sequential_3/batch_normalization_10/batchnorm/ReadVariableOpA^sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOpH^sequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOpC^sequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOpJ^sequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpE^sequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp=^sequential_3/batch_normalization_11/batchnorm/ReadVariableOpA^sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOpG^sequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOpB^sequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOpI^sequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpD^sequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp<^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp@^sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp7^sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp7^sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:::::::::::::::::::::::::::::::::::::2?
Fsequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2?
Asequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_6/AssignMovingAvg/ReadVariableOp2?
Hsequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2?
Csequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp;sequential_2/batch_normalization_6/batchnorm/ReadVariableOp2?
?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2?
Fsequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2?
Asequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_7/AssignMovingAvg/ReadVariableOp2?
Hsequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2?
Csequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp;sequential_2/batch_normalization_7/batchnorm/ReadVariableOp2?
?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2?
Fsequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2?
Asequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_8/AssignMovingAvg/ReadVariableOp2?
Hsequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2?
Csequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp;sequential_2/batch_normalization_8/batchnorm/ReadVariableOp2?
?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2\
,sequential_2/conv1d_2/BiasAdd/ReadVariableOp,sequential_2/conv1d_2/BiasAdd/ReadVariableOp2t
8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp8sequential_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2\
,sequential_2/conv1d_3/BiasAdd/ReadVariableOp,sequential_2/conv1d_3/BiasAdd/ReadVariableOp2t
8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp8sequential_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2?
Gsequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOpGsequential_3/batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOpBsequential_3/batch_normalization_10/AssignMovingAvg/ReadVariableOp2?
Isequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpIsequential_3/batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOpDsequential_3/batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2|
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp2?
@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp2?
Gsequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOpGsequential_3/batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2?
Bsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOpBsequential_3/batch_normalization_11/AssignMovingAvg/ReadVariableOp2?
Isequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpIsequential_3/batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2?
Dsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOpDsequential_3/batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2|
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp2?
@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp2?
Fsequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOpFsequential_3/batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2?
Asequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOpAsequential_3/batch_normalization_9/AssignMovingAvg/ReadVariableOp2?
Hsequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpHsequential_3/batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2?
Csequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOpCsequential_3/batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2z
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp2?
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp2p
6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp2?
Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2p
6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp2?
Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6541

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
?
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_5984

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
sequential_3_input;
$serving_default_sequential_3_input:0?????????@
sequential_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ǥ
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
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_sequential??{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_3_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_3_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}}
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
+?&call_and_return_all_conditional_losses
?__call__"?R
_tf_keras_sequential?R{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
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
+?&call_and_return_all_conditional_losses
?__call__"?P
_tf_keras_sequential?P{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
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
regularization_losses
Nmetrics
trainable_variables
	variables
Onon_trainable_variables

Players
Qlayer_metrics
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
[regularization_losses
\trainable_variables
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?


+kernel
,bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


/kernel
0bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 6]}}
?
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
}regularization_losses
~trainable_variables
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
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
regularization_losses
?metrics
trainable_variables
	variables
?non_trainable_variables
?layers
?layer_metrics
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?	

7kernel
8bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

;kernel
<bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
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
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 6]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

?kernel
@bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
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
$regularization_losses
?metrics
%trainable_variables
&	variables
?non_trainable_variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_6/kernel
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
/:-2conv1d_transpose_2/kernel
%:#2conv1d_transpose_2/bias
*:(2batch_normalization_10/gamma
):'2batch_normalization_10/beta
/:-2conv1d_transpose_3/kernel
%:#2conv1d_transpose_3/bias
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
 :2dense_7/kernel
 :2dense_4/kernel
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
%:#2conv1d_2/kernel
:2conv1d_2/bias
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
%:#2conv1d_3/kernel
:2conv1d_3/bias
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
 :2dense_5/kernel
:2dense_5/bias
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
2:0 (2"batch_normalization_10/moving_mean
6:4 (2&batch_normalization_10/moving_variance
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
Rregularization_losses
?metrics
Strainable_variables
T	variables
?non_trainable_variables
?layers
?layer_metrics
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
Wregularization_losses
?metrics
Xtrainable_variables
Y	variables
?non_trainable_variables
?layers
?layer_metrics
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
[regularization_losses
?metrics
\trainable_variables
]	variables
?non_trainable_variables
?layers
?layer_metrics
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
_regularization_losses
?metrics
`trainable_variables
a	variables
?non_trainable_variables
?layers
?layer_metrics
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
cregularization_losses
?metrics
dtrainable_variables
e	variables
?non_trainable_variables
?layers
?layer_metrics
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
hregularization_losses
?metrics
itrainable_variables
j	variables
?non_trainable_variables
?layers
?layer_metrics
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
lregularization_losses
?metrics
mtrainable_variables
n	variables
?non_trainable_variables
?layers
?layer_metrics
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
pregularization_losses
?metrics
qtrainable_variables
r	variables
?non_trainable_variables
?layers
?layer_metrics
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
uregularization_losses
?metrics
vtrainable_variables
w	variables
?non_trainable_variables
?layers
?layer_metrics
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
yregularization_losses
?metrics
ztrainable_variables
{	variables
?non_trainable_variables
?layers
?layer_metrics
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
}regularization_losses
?metrics
~trainable_variables
	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
A0
B1
C2
D3
E4
F5"
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
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
?regularization_losses
?metrics
?trainable_variables
?	variables
?non_trainable_variables
?layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
G0
H1
I2
J3
K4
L5"
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
.
A0
B1"
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
.
C0
D1"
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
.
E0
F1"
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
.
G0
H1"
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
.
I0
J1"
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
.
K0
L1"
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
?2?
__inference__wrapped_model_5274?
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
sequential_3_input?????????
?2?
F__inference_sequential_4_layer_call_and_return_conditional_losses_8405
F__inference_sequential_4_layer_call_and_return_conditional_losses_7673
F__inference_sequential_4_layer_call_and_return_conditional_losses_7593
F__inference_sequential_4_layer_call_and_return_conditional_losses_8641?
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
+__inference_sequential_4_layer_call_fn_8799
+__inference_sequential_4_layer_call_fn_8720
+__inference_sequential_4_layer_call_fn_7833
+__inference_sequential_4_layer_call_fn_7992?
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
F__inference_sequential_3_layer_call_and_return_conditional_losses_9133
F__inference_sequential_3_layer_call_and_return_conditional_losses_6033
F__inference_sequential_3_layer_call_and_return_conditional_losses_6085
F__inference_sequential_3_layer_call_and_return_conditional_losses_8990?
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
+__inference_sequential_3_layer_call_fn_6179
+__inference_sequential_3_layer_call_fn_6272
+__inference_sequential_3_layer_call_fn_9174
+__inference_sequential_3_layer_call_fn_9215?
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
F__inference_sequential_2_layer_call_and_return_conditional_losses_7095
F__inference_sequential_2_layer_call_and_return_conditional_losses_7149
F__inference_sequential_2_layer_call_and_return_conditional_losses_9457
F__inference_sequential_2_layer_call_and_return_conditional_losses_9360?
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
+__inference_sequential_2_layer_call_fn_9543
+__inference_sequential_2_layer_call_fn_9500
+__inference_sequential_2_layer_call_fn_7344
+__inference_sequential_2_layer_call_fn_7247?
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
"__inference_signature_wrapper_8073sequential_3_input"?
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
A__inference_dense_6_layer_call_and_return_conditional_losses_9550?
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
&__inference_dense_6_layer_call_fn_9557?
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
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9613
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9593?
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
4__inference_batch_normalization_9_layer_call_fn_9639
4__inference_batch_normalization_9_layer_call_fn_9626?
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
A__inference_re_lu_3_layer_call_and_return_conditional_losses_9644?
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
&__inference_re_lu_3_layer_call_fn_9649?
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
C__inference_reshape_3_layer_call_and_return_conditional_losses_9662?
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
(__inference_reshape_3_layer_call_fn_9667?
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
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_5454?
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
1__inference_conv1d_transpose_2_layer_call_fn_5464?
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
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9723
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9703?
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
5__inference_batch_normalization_10_layer_call_fn_9736
5__inference_batch_normalization_10_layer_call_fn_9749?
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
A__inference_re_lu_4_layer_call_and_return_conditional_losses_9754?
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
&__inference_re_lu_4_layer_call_fn_9759?
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
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_5644?
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
1__inference_conv1d_transpose_3_layer_call_fn_5654?
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
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9795
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9815?
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
5__inference_batch_normalization_11_layer_call_fn_9828
5__inference_batch_normalization_11_layer_call_fn_9841?
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
A__inference_re_lu_5_layer_call_and_return_conditional_losses_9846?
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
&__inference_re_lu_5_layer_call_fn_9851?
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
C__inference_flatten_3_layer_call_and_return_conditional_losses_9863?
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
(__inference_flatten_3_layer_call_fn_9868?
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
A__inference_dense_7_layer_call_and_return_conditional_losses_9876?
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
&__inference_dense_7_layer_call_fn_9883?
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
A__inference_dense_4_layer_call_and_return_conditional_losses_9890?
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
&__inference_dense_4_layer_call_fn_9897?
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
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9953
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9933?
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
4__inference_batch_normalization_6_layer_call_fn_9966
4__inference_batch_normalization_6_layer_call_fn_9979?
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
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_9984?
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
,__inference_leaky_re_lu_3_layer_call_fn_9989?
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
D__inference_reshape_2_layer_call_and_return_conditional_losses_10002?
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
)__inference_reshape_2_layer_call_fn_10007?
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
C__inference_conv1d_2_layer_call_and_return_conditional_losses_10022?
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
(__inference_conv1d_2_layer_call_fn_10031?
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
?2?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10087
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10067
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10169
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10149?
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
?2?
5__inference_batch_normalization_7_layer_call_fn_10182
5__inference_batch_normalization_7_layer_call_fn_10113
5__inference_batch_normalization_7_layer_call_fn_10195
5__inference_batch_normalization_7_layer_call_fn_10100?
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
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_10200?
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
-__inference_leaky_re_lu_4_layer_call_fn_10205?
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
C__inference_conv1d_3_layer_call_and_return_conditional_losses_10220?
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
(__inference_conv1d_3_layer_call_fn_10229?
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
?2?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10265
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10285
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10367
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10347?
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
?2?
5__inference_batch_normalization_8_layer_call_fn_10380
5__inference_batch_normalization_8_layer_call_fn_10311
5__inference_batch_normalization_8_layer_call_fn_10393
5__inference_batch_normalization_8_layer_call_fn_10298?
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
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_10398?
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
-__inference_leaky_re_lu_5_layer_call_fn_10403?
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_10409?
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
)__inference_flatten_2_layer_call_fn_10414?
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
B__inference_dense_5_layer_call_and_return_conditional_losses_10424?
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
'__inference_dense_5_layer_call_fn_10433?
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
__inference__wrapped_model_5274?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@;?8
1?.
,?)
sequential_3_input?????????
? ";?8
6
sequential_2&?#
sequential_2??????????
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9703|CD-.@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_9723|D-C.@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
5__inference_batch_normalization_10_layer_call_fn_9736oCD-.@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
5__inference_batch_normalization_10_layer_call_fn_9749oD-C.@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9795|EF12@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_9815|F1E2@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
5__inference_batch_normalization_11_layer_call_fn_9828oEF12@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
5__inference_batch_normalization_11_layer_call_fn_9841oF1E2@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9933bGH563?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9953bH5G63?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
4__inference_batch_normalization_6_layer_call_fn_9966UGH563?0
)?&
 ?
inputs?????????
p
? "???????????
4__inference_batch_normalization_6_layer_call_fn_9979UH5G63?0
)?&
 ?
inputs?????????
p 
? "???????????
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10067|IJ9:@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10087|J9I:@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10149jIJ9:7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10169jJ9I:7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
5__inference_batch_normalization_7_layer_call_fn_10100oIJ9:@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
5__inference_batch_normalization_7_layer_call_fn_10113oJ9I:@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
5__inference_batch_normalization_7_layer_call_fn_10182]IJ9:7?4
-?*
$?!
inputs?????????
p
? "???????????
5__inference_batch_normalization_7_layer_call_fn_10195]J9I:7?4
-?*
$?!
inputs?????????
p 
? "???????????
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10265|KL=>@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10285|L=K>@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10347jKL=>7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10367jL=K>7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
5__inference_batch_normalization_8_layer_call_fn_10298oKL=>@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
5__inference_batch_normalization_8_layer_call_fn_10311oL=K>@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
5__inference_batch_normalization_8_layer_call_fn_10380]KL=>7?4
-?*
$?!
inputs?????????
p
? "???????????
5__inference_batch_normalization_8_layer_call_fn_10393]L=K>7?4
-?*
$?!
inputs?????????
p 
? "???????????
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9593bAB)*3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_9613bB)A*3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
4__inference_batch_normalization_9_layer_call_fn_9626UAB)*3?0
)?&
 ?
inputs?????????
p
? "???????????
4__inference_batch_normalization_9_layer_call_fn_9639UB)A*3?0
)?&
 ?
inputs?????????
p 
? "???????????
C__inference_conv1d_2_layer_call_and_return_conditional_losses_10022d783?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
(__inference_conv1d_2_layer_call_fn_10031W783?0
)?&
$?!
inputs?????????
? "???????????
C__inference_conv1d_3_layer_call_and_return_conditional_losses_10220d;<3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
(__inference_conv1d_3_layer_call_fn_10229W;<3?0
)?&
$?!
inputs?????????
? "???????????
L__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_5454v+,<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
1__inference_conv1d_transpose_2_layer_call_fn_5464i+,<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
L__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_5644v/0<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
1__inference_conv1d_transpose_3_layer_call_fn_5654i/0<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
A__inference_dense_4_layer_call_and_return_conditional_losses_9890[4/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
&__inference_dense_4_layer_call_fn_9897N4/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_5_layer_call_and_return_conditional_losses_10424\?@/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_5_layer_call_fn_10433O?@/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_6_layer_call_and_return_conditional_losses_9550[(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
&__inference_dense_6_layer_call_fn_9557N(/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_7_layer_call_and_return_conditional_losses_9876d38?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
&__inference_dense_7_layer_call_fn_9883W38?5
.?+
)?&
inputs??????????????????
? "???????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_10409\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? |
)__inference_flatten_2_layer_call_fn_10414O3?0
)?&
$?!
inputs?????????
? "???????????
C__inference_flatten_3_layer_call_and_return_conditional_losses_9863n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
(__inference_flatten_3_layer_call_fn_9868a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
G__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_9984X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_leaky_re_lu_3_layer_call_fn_9989K/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_10200`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
-__inference_leaky_re_lu_4_layer_call_fn_10205S3?0
)?&
$?!
inputs?????????
? "???????????
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_10398`3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
-__inference_leaky_re_lu_5_layer_call_fn_10403S3?0
)?&
$?!
inputs?????????
? "???????????
A__inference_re_lu_3_layer_call_and_return_conditional_losses_9644X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? u
&__inference_re_lu_3_layer_call_fn_9649K/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_re_lu_4_layer_call_and_return_conditional_losses_9754r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
&__inference_re_lu_4_layer_call_fn_9759e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
A__inference_re_lu_5_layer_call_and_return_conditional_losses_9846r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
&__inference_re_lu_5_layer_call_fn_9851e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
D__inference_reshape_2_layer_call_and_return_conditional_losses_10002\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? |
)__inference_reshape_2_layer_call_fn_10007O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_reshape_3_layer_call_and_return_conditional_losses_9662\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? {
(__inference_reshape_3_layer_call_fn_9667O/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_sequential_2_layer_call_and_return_conditional_losses_7095|4GH5678IJ9:;<KL=>?@>?;
4?1
'?$
dense_4_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_7149|4H5G678J9I:;<L=K>?@>?;
4?1
'?$
dense_4_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_9360u4GH5678IJ9:;<KL=>?@7?4
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
F__inference_sequential_2_layer_call_and_return_conditional_losses_9457u4H5G678J9I:;<L=K>?@7?4
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
+__inference_sequential_2_layer_call_fn_7247o4GH5678IJ9:;<KL=>?@>?;
4?1
'?$
dense_4_input?????????
p

 
? "???????????
+__inference_sequential_2_layer_call_fn_7344o4H5G678J9I:;<L=K>?@>?;
4?1
'?$
dense_4_input?????????
p 

 
? "???????????
+__inference_sequential_2_layer_call_fn_9500h4GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "???????????
+__inference_sequential_2_layer_call_fn_9543h4H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
F__inference_sequential_3_layer_call_and_return_conditional_losses_6033{(AB)*+,CD-./0EF123>?;
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
F__inference_sequential_3_layer_call_and_return_conditional_losses_6085{(B)A*+,D-C./0F1E23>?;
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
F__inference_sequential_3_layer_call_and_return_conditional_losses_8990t(AB)*+,CD-./0EF1237?4
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
F__inference_sequential_3_layer_call_and_return_conditional_losses_9133t(B)A*+,D-C./0F1E237?4
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
+__inference_sequential_3_layer_call_fn_6179n(AB)*+,CD-./0EF123>?;
4?1
'?$
dense_6_input?????????
p

 
? "???????????
+__inference_sequential_3_layer_call_fn_6272n(B)A*+,D-C./0F1E23>?;
4?1
'?$
dense_6_input?????????
p 

 
? "???????????
+__inference_sequential_3_layer_call_fn_9174g(AB)*+,CD-./0EF1237?4
-?*
 ?
inputs?????????
p

 
? "???????????
+__inference_sequential_3_layer_call_fn_9215g(B)A*+,D-C./0F1E237?4
-?*
 ?
inputs?????????
p 

 
? "???????????
F__inference_sequential_4_layer_call_and_return_conditional_losses_7593?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@C?@
9?6
,?)
sequential_3_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_4_layer_call_and_return_conditional_losses_7673?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@C?@
9?6
,?)
sequential_3_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_4_layer_call_and_return_conditional_losses_8405?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@7?4
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
F__inference_sequential_4_layer_call_and_return_conditional_losses_8641?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@7?4
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
+__inference_sequential_4_layer_call_fn_7833?%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@C?@
9?6
,?)
sequential_3_input?????????
p

 
? "???????????
+__inference_sequential_4_layer_call_fn_7992?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@C?@
9?6
,?)
sequential_3_input?????????
p 

 
? "???????????
+__inference_sequential_4_layer_call_fn_8720z%(AB)*+,CD-./0EF1234GH5678IJ9:;<KL=>?@7?4
-?*
 ?
inputs?????????
p

 
? "???????????
+__inference_sequential_4_layer_call_fn_8799z%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
"__inference_signature_wrapper_8073?%(B)A*+,D-C./0F1E234H5G678J9I:;<L=K>?@Q?N
? 
G?D
B
sequential_3_input,?)
sequential_3_input?????????";?8
6
sequential_2&?#
sequential_2?????????