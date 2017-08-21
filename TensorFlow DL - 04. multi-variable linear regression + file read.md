<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 04-1. Multi-variable linear regression

x 많으면 더 잘 예측할 수 있을 것

H(x1, x2, x3) = x1w1 + x2w2 + x3w3

참조 링크  
https://www.youtube.com/watch?v=fZUV3xjoZSM&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multi-variable linear regression

```
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
x1_data = [ 73.,  93.,  89.,  96.,  73.]
x2_data = [ 80.,  88.,  91.,  98.,  66.]
x3_data = [ 75.,  93.,  90., 100.,  70.]
y_data  = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y  = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b  = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
h = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost function
cost = tf.reduce_mean(tf.square(h - Y))

# Minimize: small learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op)
sess = tf.Session()                         # launch the graph in a session
sess.run(tf.global_variables_initializer()) # init global variables


# 3) Update variables in the graph (and return values)
for step in range(2001):
	cost_val, h_val, _ = sess.run([cost, h, train], 
	                     feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```
<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-1-multi_variable_linear_regression.py  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multi-variable linear regression using matrix

H(X) = XW + b

```
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
          
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
h = tf.matmul(X, W) + b   # matmul (5 x 3) * (3 x 1)

# cost function
cost = tf.reduce_mean(tf.square(h - Y))

# Minimize: small learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op)
sess = tf.Session()                         # launch the graph in a session
sess.run(tf.global_variables_initializer()) # init global variables


# 3) Update variables in the graph (and return values)
for step in range(2001):
	cost_val, h_val, _ = sess.run([cost, h, train], 
	                     feed_dict={X: x_data, Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```
<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-2-multi_variable_matmul_linear_regression.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 04-2. Loading data from file

참조 링크  
https://www.youtube.com/watch?v=o2q4QNnoShY&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Python: Slicing

```
nums = range(5)     # range: list of integers
print(nums)
print(nums[2:4])    # slice [2, 4)
print(nums[2: ])    # slice [2, 5)
print(nums[ :2])    # slice [start, 2)
print(nums[ : ])    # [0, 5)
print(nums[:-1])    # [0, 5 - 1)
print(nums[:-2])    # [0, 5 - 2)
nums[2:4] = [8, 9]  # num[2, 4) <- {"8, 9"}
print(nums)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Numpy

Array를 index, slice, iterate 처럼 처리 가능  
: 대신 ... 사용 가능

```
import numpy as np

# 1D array: shape[3]
a1 = np.array( [1, 2, 3] )
print(a1.ndim, a1.shape, a1.dtype)

# 1D array slicing
               # index   = result
print(a1)      # [0, 3)  = [1,2,3]
print(a1[1:3]) # [1, 3)  = [2,3]
print(a1[-1])  # [3 - 1) = 3
print(a1[0:2]) # [0: 2)  = [1 2] 


# 2D array: shape[1, 3]
a2 = np.array([ [1, 2, 3] ])
print(a2.ndim, a2.shape, a2.dtype)

# 2D array: shape[3, 1]
a3 = np.array([ [1], [2], [3] ])
print(a3.ndim, a3.shape, a3.dtype)

# 2D array: shape[3, 4]
b = np.array([ [1,2,3,4],
               [5,6,7,8],
               [9,10,11,12] ])
print(b.ndim, b.shape, b.dtype)

# 2D array slicing
                   # index                   = result
print(b[ :, 1])    # row [0, 3) , col [1)    = [ 2  6 10]
print(b[-1])       # row [3 - 1)             = [ 9 10 11 12]
print(b[-1,...])   # row [3 - 1), col [0, 4) = [ 9 10 11 12]
print(b[0:2, :])   # row [0, 2) , col [0, 4) = [[1 2 3 4]  [5 6 7 8]]


# ndarray 생성 + 초기화
c = np.random.randn(2,3)
print(c.ndim, c.shape, c.dtype, c)

print(
np.empty((2,3)),
np.ones((2,3)),
np.zeros((2,3)),
np.arange(15))
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Loading data from file

테스트 데이터 파일은  
C:\Users\(사용자 계정이름)\Docker\work 에 저장해서 사용

data01_test_score.csv 파일 내용

```
# x1: EXAM1 점수, x2: EXAM2 점수, x3: EXAM3 점수, y: FINAL 점수

73,80,75,152
93,88,93,185
89,91,90,180
96,98,100,196
73,66,70,142
53,46,55,101
```

python 코드
```
import numpy as np

import os
os.getcwd()
os.chdir('/home/testu/work') 
os.getcwd()

xy = np.loadtxt('data01_test_score.csv', delimiter=',', dtype=np.float32)
print(xy.ndim, xy.shape, xy, len(xy))


x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.ndim, x_data.shape, x_data, len(x_data))
print(y_data.ndim, y_data.shape, y_data, len(y_data))


y_data2 = xy[:, -1]
print(y_data2.ndim, y_data2.shape, y_data2, len(y_data2))
```

실행 결과
```
(2, (6, 3), array([[  73.,   80.,   75.],
       [  93.,   88.,   93.],
       [  89.,   91.,   90.],
       [  96.,   98.,  100.],
       [  73.,   66.,   70.],
       [  53.,   46.,   55.]], dtype=float32), 6)
(2, (6, 1), array([[ 152.],
       [ 185.],
       [ 180.],
       [ 196.],
       [ 142.],
       [ 101.]], dtype=float32), 6)
(1, (6,), array([ 152.,  185.,  180.,  196.,  142.,  101.], dtype=float32), 6)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multi-variable linear regression with csv

```
import tensorflow as tf
import numpy as np

import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
xy = np.loadtxt('data01_test_score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
h = tf.matmul(X, W) + b   # matmul (6 x 3) * (3 x 1)

# cost function
cost = tf.reduce_mean(tf.square(h - Y))

# Minimize: small learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op)
sess = tf.Session()                         # launch the graph in a session
sess.run(tf.global_variables_initializer()) # init global variables


# 3) Update variables in the graph (and return values)
for step in range(2001):
	cost_val, h_val, _ = sess.run([cost, h, train], 
	                     feed_dict={X: x_data, Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)

print("other score test1 ", sess.run(h, feed_dict={X: [[100, 70, 101]]}))
print("other score test2 ", sess.run(h, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Thread and Queues

파일 커서 메모리에 한 번에 안 올라가는 경우  
여러개 파일 읽어서 처리할 경우  

files -> (random shuffle) -> filename queue 에 쌓아 -> (Reader .. -> Decoder ..) -> Example queue (여기서 꺼내 써)

1) 읽을 파일 리스트  
filename_queue = tf.train.string_input_producer(['a.csv', 'b.csv', ...], shuffle=False, name='filenamequeue')

2) 파일 읽어올 reader 정의  
reader = tf.TextLineReader()  
key, value = reader.read(filename_queue)

3) decoder 정의  
record_defaults = [[0.], [0.], [0.], [0.]] # 읽을 데이터 타입 정의 (여기선 float)  
xy = tf.decode_csv(value, record_defaults=record_defaults)

다음과 같은 형태로 사용
```
# tf.train.batch
# collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # x_data, y_data, 10개씩 가져와

sess = tf.Session()

...

# filename queue 관리
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
	x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
	...

coord.request_stop()
coord.join(threads)
```
<br />

참조 링크  
https://www.tensorflow.org/programmers_guide/threading_and_queues#queuerunner


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multi-variable linear regression with Queues


```
import tensorflow as tf
import numpy as np
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations

filename_queue = tf.train.string_input_producer(
	['data_01_test_score.csv'], shuffle = False, name = 'filename_queue')

reader     = tf.TextLineReader()
key, value = reader.read(filename_queue)

# default values (for empty columns), type of decoded results
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)

# collect batches of csv
train_x_batch, train_y_batch = tf.train.batch(
	[xy[0:-1], xy[-1:]], batch_size = 10)

# old code
# xy = np.loadtxt('data01_test_score.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1])   , name=  'bias'  )

# hypothesis
h = tf.matmul(X, W) + b   # matmul (6 x 3) * (3 x 1)

# cost function
cost = tf.reduce_mean(tf.square(h - Y))

# Minimize: small learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train     = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op)
sess = tf.Session()                         # launch the graph in a session
sess.run(tf.global_variables_initializer()) # init global variables

# Start populating the filename queue 
coord   = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

# 3) Update variables in the graph (and return values)
for step in range(2001):
	x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
	cost_val, h_val, _ = sess.run([cost, h, train], 
	                     feed_dict={X: x_batch, Y: y_batch})
	if step % 10 == 0:
		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)

coord.request_stop()
coord.join(threads)

print("other score test1 ", sess.run(h, feed_dict={X: [[100, 70, 101]]}))
print("other score test2 ", sess.run(h, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-4-tf_reader_linear_regression.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### shuffle_batch

tf.train.batch 대신 배치 순서 섞고 싶을 때 사용

```
# 랜덤하게 샘플할 버퍼 크기
# 크면 잘 섞이지만 느리고 메모리 많이 사용함
min_after_dequeue = 10000

# capacity는 min_after_dequeue 보다 커야함, prefetch할 최대 크기 결정
# 보통 min_after_dequeue + (thread 개수 + 작은 safety margin) * batch_size
capacity = min_after_dequeue + 3 * batch_size

example_batch, label_batch = tf.train.shuffle_batch(
	[example, label], batch_size = batch_size, capacity = capacity,
	min_after_dequeue = min_after_dequeue)
```

