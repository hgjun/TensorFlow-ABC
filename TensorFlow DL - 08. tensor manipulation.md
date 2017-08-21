<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## Lab08. Tensor Manipulation

<br />
���� ��ũ  
https://www.youtube.com/watch?v=ZYX0FaqUeN4&feature=youtu.be  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-08-tensor_manipulation.ipynb

������ ���� �ڵ�

```
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Simple array

```
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t)

print("Rank: " , t.ndim)
print("Shape: ", t.shape) # 7 �� elements

# Slicing
print(t[0], t[1], t[-1])  # [0], [1], [7 - 1]
print(t[2:5], t[4:-1])    # [2, 5), [4, 7 - 1)
print(t[:2], t[3:])       # [0, 2), [3, 7)

# array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.])
# [ 0.  1.  2.  3.  4.  5.  6.]
# ('Rank: ', 1)
# ('Shape: ', (7,))
# (0.0, 1.0, 6.0)
# (array([ 2.,  3.,  4.]), array([ 4.,  5.]))
# (array([ 0.,  1.]), array([ 3.,  4.,  5.,  6.]))
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 2D array

```
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print("Rank: {}, Shape: {}" .format(t.ndim, t.shape))

# array([[  1.,   2.,   3.],
#        [  4.,   5.,   6.],
#        [  7.,   8.,   9.],
#        [ 10.,  11.,  12.]])
# Rank: 2, Shape: (4, 3)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Shape, Rank, Axis

```
# Rank = square bracket ����
# Shape = �� ������ ������ element �������� �����
 
t = tf.constant([1,2,3,4])      # 1����, Rank 1
tf.shape(t).eval()
# array([4], dtype=int32)


t = tf.constant([[1,2], [3,4]]) # Rank 2
tf.shape(t).eval()
# array([2, 2], dtype=int32)


t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()
# array([1, 2, 3, 4], dtype=int32)

# '[' 4�� -> Rank 4
# shape �����ͺ��� ���� 4, 3, 2, 1 -> Shape [1, 2, 3, 4]
# Axis ��

[ Axis 0
	[  Axis 1
		[  Axis 2
			[1, 2, 3, 4],  Axis 3 �̸鼭 Axis -1 (���� ����, �� �������� �ִ� ��)
			[5, 6, 7, 8],
			[9, 10, 11, 12]
		],
		[
			[13, 14, 15, 16],
			[17, 18, 19, 20],
			[21, 22, 23, 24]
		]
	]
]
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Matmul VS multiply

```
matrix1 = tf.constant([[1., 2.], [3., 4.]])   # Rank 2, Shape (2, 2)
matrix2 = tf.constant([[1.],[2.]])            # Rank 2, Shape (2, 1)
tf.matmul(matrix1, matrix2).eval()            # [2 x 2] * [2 x 1] = [2 x 1]
# array([[  5.],
#       [ 11.]], dtype=float32)

(matrix1 * matrix2).eval()   # ��� �� �ƴ�!
# array([[ 1.,  2.],
#       [ 6.,  8.]], dtype=float32)


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
tf.matmul(matrix1, matrix2).eval()
# array([[ 12.]], dtype=float32)
# [1 x 2] x [2 x 1] = [1 x 1] matrix

a = [[3., 3.]]
b = [[2.],[2.]]
np.matmul(a, b)
# array([[ 12.]])


(matrix1*matrix2).eval()  # ��� �� �ƴ�!
# array([[ 6.,  6.],
#       [ 6.,  6.]], dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Broadcasting (����!)

Operations between the same shapes  
������ ���� ������, �ſ� �����ؼ� ����ؾ�!

```
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1 + matrix2).eval() 
# array([[ 5.,  5.]], dtype=float32)

(matrix1 * matrix2).eval() 
# array([[ 6.,  6.]], dtype=float32)


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()
# array([[ 5.,  5.],
#       [ 5.,  5.]], dtype=float32)

   
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()
# array([[ 5.,  5.]], dtype=float32)


Shape �޶� ���갡��
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)  # -> [[3, 3]] ���� �����༭ ���
(matrix1 + matrix2).eval()
# array([[ 4.,  5.], dtype=float32)

Rank �޶� ���갡��
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
(matrix1 + matrix2).eval()
# array([[ 4.,  6.], dtype=float32)


matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
(matrix1 + matrix2).eval()
# array([[ 4.,  5.],
#       [ 5.,  6.]], dtype=float32)
```

<br />

���� ��ũ  
https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html 


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Random values for variable initializations

```
tf.random_normal([3]).eval()
# array([ 0.93376452, -0.45112884, -0.56874341], dtype=float32)

tf.random_uniform([2]).eval()
# array([ 0.13913763,  0.06281352], dtype=float32)

tf.random_uniform([2, 3]).eval()
# array([[ 0.08339167,  0.35136676,  0.45676231],
#        [ 0.10042191,  0.32939088,  0.49997854]], dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Reduce Mean/Sum

```
tf.reduce_mean([1, 2], axis=0).eval()  # data type�� int �� ��� 1 ���°�!
# 1

tf.reduce_mean([1., 2.], axis=0).eval()
# 1.5


x = [[1., 2.], [3., 4.]]  # Rank 2, Shape (2, 2)
# [  ��Axis 0
#     [1., 2.], �� Axis 1 �̸鼭 Axis -1
#     [3., 4.]
# ]

tf.reduce_mean(x).eval()
# 2.5

tf.reduce_mean(x, axis=0).eval()  # Axis 0 ���� ��� (1, 3) (2, 4)
# array([ 2.,  3.], dtype=float32)

tf.reduce_mean(x, axis=1).eval()  # Axis 1 ���� ��� (1, 2) (3, 4)
# array([ 1.5,  3.5], dtype=float32)

tf.reduce_mean(x, axis=-1).eval() # �� ����� Axis 1 = Axis -1 
# array([ 1.5,  3.5], dtype=float32)



tf.reduce_sum(x).eval()
# 10.0

tf.reduce_sum(x, axis=0).eval()   # Axis 0 ���� �� (1, 3) (2, 4)
# array([ 4.,  6.], dtype=float32)

tf.reduce_sum(x, axis=-1).eval()  # Axis 1 ���� �� (1, 2) (3, 4)
# array([ 3.,  7.], dtype=float32)

# ���� ���̴� ����
# ���� ���ʿ� �ִ� ���� ��ҵ� ���ļ� ��ճ���
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()
# 5.0
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Argmax with axis

```
x = [[0, 1, 2], [2, 1, 0]]
# [   ��Axis 0
#     [0, 1, 2], �� Axis 1 �̸鼭 Axis -1
#     [2, 1, 0]
# ]

tf.argmax(x, axis=0).eval()   # Axis 0 ���� argmax (0, 2) (1, 1)  (2, 0)
# array([1, 0, 0])            # index ���� �ǳ���

tf.argmax(x, axis=1).eval()   # Axis 1 ���� argmax (0, 1, 2) (2, 1, 0)
# array([2, 0])

tf.argmax(x, axis=-1).eval()  # �� ����� Axis 1 = Axis -1 
# array([2, 0])
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Reshape!!

```
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])

# "[" 3 ���� Rank 3

t.shape
# (2, 2, 3)  �� �� "[ ]" ���� ��� ���� ī��Ʈ�ϸ� ���� 3, 2, 2

# Rank 2 �� reshape
# ���� �ȿ� ���� 3���� �ϰ� �������� �˾Ƽ� ó��(-1)�ض�
tf.reshape(t, shape=[-1, 3]).eval()
# array([[ 0,  1,  2],
#        [ 3,  4,  5],
#        [ 6,  7,  8],
#        [ 9, 10, 11]])

# Rank 3 ���� reshape
# ������ 3 �״��, �� ������ 1, ������ �˾Ƽ�
tf.reshape(t, shape=[-1, 1, 3]).eval()
# array([[[ 0,  1,  2]],
#        [[ 3,  4,  5]],
#        [[ 6,  7,  8]],
#        [[ 9, 10, 11]]])
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Reshape: squeeze, expand

������ ����
```
tf.squeeze([[0], [1], [2]]).eval()
# array([0, 1, 2], dtype=int32)
```

���� �߰��ϰ� ���� ��  
tensor�� shape �����Ű�� ���� ���
```
tf.expand_dims([0, 1, 2], 1).eval()
# array([[0],
#        [1],
#       [2]], dtype=int32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### One hot

```
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
# one_hot()�� rank �ϳ� �÷�  ���� rank 2 ������ rank 3 ��
# array([[[ 1.,  0.,  0.]],
#        [[ 0.,  1.,  0.]],
#        [[ 0.,  0.,  1.]],
#        [[ 1.,  0.,  0.]]], dtype=float32)

t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()
# reshape���� �ϳ� ���� rank ���� �� �־�
# array([[ 1.,  0.,  0.],
#        [ 0.,  1.,  0.],
#        [ 0.,  0.,  1.],
#        [ 1.,  0.,  0.]], dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Casting

������ Ÿ�� ����

```
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
# array([1, 2, 3, 4], dtype=int32)

tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()
# sum �ؼ� true ���� ���� �� ����
# array([1, 0, 1, 0], dtype=int32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Stack

```
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=0).eval()
# array([[1, 4],
#        [2, 5],
#        [3, 6]], dtype=int32)


tf.stack([x, y, z], axis=1).eval()
tf.stack([x, y, z], axis=-1).eval()
# array([[1, 2, 3],
#        [4, 5, 6]], dtype=int32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Ones like and Zeros like

```
x = [[0, 1, 2],
     [2, 1, 0]]

x�� ���� ������ ����� 1 ä����
tf.ones_like(x).eval()
# array([[1, 1, 1],
#        [1, 1, 1]], dtype=int32)

x�� ���� ������ ����� 0 ä����
tf.zeros_like(x).eval()
# array([[0, 0, 0],
#        [0, 0, 0]], dtype=int32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Zip

```
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
# (1, 4)
# (2, 5)
# (3, 6)


for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
# (1, 4, 7)
# (2, 5, 8)
# (3, 6, 9)


zip �Լ��� ������ ������ ��Ұ��� ���� ������ �ڷ����� �����ִ� ����
print(zip([1,2,3], [4,5,6]))
[(1, 4), (2, 5), (3, 6)]

print(zip([1,2,3], [4,5,6], [7,8,9]))
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

print(zip("abc", "def"))
[('a', 'd'), ('b', 'e'), ('c', 'f')]

# for zip ����1
a = [1,2,3,4,5]
b = ['a','b','c','d','e']
 
for x,y in zip (a,b):
  print(x,y)
# 1 a
# 2 b
# 3 c
# 4 d 
# 5 e

  
# for zip ����2
t = np.array( [[0], [3]])
print(t)            #  [[0], [3]]
print(t.flatten())  # [0 3]

v = [1,2]
for p, r in zip(v, t.flatten()):
print(p, r)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Transpose

```
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)
# (2, 2, 3)
# array([[
#           [ 0,  1,  2],
#           [ 3,  4,  5]
#        ],
#        [
#           [ 6,  7,  8],
#           [ 9, 10, 11]
#        ]])
```

t�� axis 0 1 2 ��  axis 1 0 2 �� �ٲٰڴ�
```
axis 0  [0, 1, 2], [6, 7, 8]
        [3, 4, 5], [9, 10, 11]
        
axis 1  [[0, 1, 2], [3, 4, 5]]
axis 1  [[6, 7, 8], [9, 10, 11]]
(axis 1 ������ �� ������ axis 0 ��)

axis 2  [0, 1, 2], ...
```
```
t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))
# (2, 2, 3)
# array([[
#           [ 0,  1,  2],
#           [ 6,  7,  8]
#        ],
#        [
#           [ 3,  4,  5],
#           [ 9, 10, 11]
#        ]])
```

t1�� axis 0 1 2 ��  axis 1 0 2 �� �ٲٰڴ�
```
axis 0  [0, 1, 2], [3, 4, 5]
        [6, 7, 8], [9, 10, 11]
        
axis 1  [[0, 1, 2], [6, 7, 8]]
axis 1  [[3, 4, 5], [9, 10, 11]]
(axis 1 ������ �� ������ axis 0 ��)

axis 2  [0, 1, 2], ...
```
```
t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
# (2, 2, 3)
# array([[
#          [ 0,  1,  2],
#          [ 3,  4,  5]
#        ],
#        [
#          [ 6,  7,  8],
#          [ 9, 10, 11]
#        ]])
```


t�� axis 0 1 2 ��  axis 1 2 0 ���� �ٲٰڴ�
```
shape (2, 2, 3) -> shape (2, 3, 2)

axis 0  [0, 1, 2], [6, 7, 8]
        [3, 4, 5], [9, 10, 11]
        
axis 1  [[0, 1, 2], [3, 4, 5]]
axis 1  [[6, 7, 8], [9, 10, 11]]
(axis 1 ������ �� ������ axis 0 ��)

axis 2  [0, 1, 2], ...
```

```
0 (����axis 1)  [0, 1, 2], [3, 4, 5]
0 (����axis 1)  [6, 7, 8], [9, 10, 11]

1 (����axis 2)  [ 0,  6], [ 1,  7], [ 2,  8]
1 (����axis 2)  [ 3,  9], [ 4, 10], [ 5, 11]
```
```
t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))
# (2, 3, 2)
# array([[
#           [ 0,  6],
#           [ 1,  7],
#           [ 2,  8]
#        ],
#        [
#           [ 3,  9],
#           [ 4, 10],
#           [ 5, 11]
#        ]])
```


```
t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
# (2, 2, 3)
# array([[
#           [ 0,  1,  2],
#           [ 3,  4,  5]
#        ],
#        [
#           [ 6,  7,  8],
#           [ 9, 10, 11]
#        ]])
```


