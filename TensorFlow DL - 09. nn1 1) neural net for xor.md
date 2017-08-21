<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 09-1. Neural Net for XOR

<br />  
#### XOR

| X1 | X2 | Y  |
|:--:|:--:|:--:|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |


Y 값 binary 이니 binary logistic regression 사용하면 됨

참조 링크  
https://www.youtube.com/watch?v=oFGHOsAYiz0&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### XOR problem

Linear regression 으로는 잘 분류 안돼

```
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

# 1) Build a graph using tf operations
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
          
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)


# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, [None, 2])  # X 속성 개수 2 개
Y = tf.placeholder(tf.float32, [None, 1])  # Y 속성 개수 1개

W = tf.Variable(tf.random_normal([2, 1]), name='weight')  # X feature 개수(2), Y 개수(1) -> shape[2, 1] 2개 들어와서 1개 나와
b = tf.Variable(tf.random_normal([1]), name='bias')  # 나가는 값 (Y)의 개수


# hypothesis H(X) = 1 / (1 + e^(-Σ(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)


# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))


# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)


# Accuracy computation
prediction         = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
correct_prediction = tf.equal(prediction, Y)
accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		sess.run([train], feed_dict = {X: x_data, Y: y_data})
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  형태로 출력
	
	# Accuracy report
	# hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, prediction, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```

실행 결과 (no good)
```
Hypothesis: 0.5, 0.5, 0.5, 0.5
Correct   : 0.0, 0.0, 0.0, 0.0
Accuracy  : 0.5
```

참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-1-xor.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Neural Net

```
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

# 1) Build a graph using tf operations
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
          
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)


# placeholders for a tensor that will be always fed

# Input
X = tf.placeholder(tf.float32, [None, 2])  # X 속성 개수 2 개
Y = tf.placeholder(tf.float32, [None, 1])  # Y 속성 개수 1개

# Layer 1
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')  # X feature 개수(2), Layer 개수(2) -> shape[2, 2] 2개 들어와서 2개 나와
b1 = tf.Variable(tf.random_normal([2])   , name='bias1'  )  # Layer 2 개수(2)
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# Layer 2
W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')  # 입력 개수(2), Y 개수 (2) -> shape[2, 1] 2개 들어와서 1개 나와
b2 = tf.Variable(tf.random_normal([1])   , name='bias2'  )  # Y 개수 (2)
h = tf.sigmoid(tf.matmul(layer1, W2) + b2)


# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))


# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


# Accuracy computation
prediction         = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
correct_prediction = tf.equal(prediction, Y)
accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		sess.run([train], feed_dict = {X: x_data, Y: y_data})
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  형태로 출력
	
	# Accuracy report
	# hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, prediction, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)

실행 결과
('\nHypothesis: ', array([[ 0.01338218],
       [ 0.98166394],
       [ 0.98809403],
       [ 0.01135799]], dtype=float32), '\\Correct: ', array([[ 0.],
       [ 1.],
       [ 1.],
       [ 0.]], dtype=float32), '\\Accuracy: ', 1.0)
```


참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-2-xor-nn.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Deep/Wide Neural Net for XOR

```
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

# 1) Build a graph using tf operations
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
          
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)


# placeholders for a tensor that will be always fed

# Input
X = tf.placeholder(tf.float32, [None, 2])  # X 속성 개수 2 개
Y = tf.placeholder(tf.float32, [None, 1])  # Y 속성 개수 1개

# Deep NN (Layer 층 늘어남 2 -> 4)
# Wide NN (Layer 안의 Unit 늘어남 2 -> 10)
# Layer 1
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')  # X feature 개수(2), Layer 개수(10) -> shape[2, 10] 2개 들어와서 10개 나와
b1 = tf.Variable(tf.random_normal([10])   , name='bias1'  )  # Layer 2 개수(10)
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# Layer 2
W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')  # 입력 개수(10), 출력 개수 (10) -> shape[10, 10]
b2 = tf.Variable(tf.random_normal([10])    , name='bias2'  )  # 출력 개수(10)
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# Layer 3
W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')  # 입력 개수(10), 출력 개수 (10) -> shape[10, 10]
b3 = tf.Variable(tf.random_normal([10])    , name='bias3'  )  # 출력 개수(10)
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

# Layer 4
W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')  # 입력 개수(10), 출력 개수 (1) -> shape[10, 1]
b4 = tf.Variable(tf.random_normal([1])    , name='bias4'  )  # Y 개수(1)
h = tf.sigmoid(tf.matmul(layer3, W4) + b4)



# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))


# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


# Accuracy computation
prediction         = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
correct_prediction = tf.equal(prediction, Y)
accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		sess.run([train], feed_dict = {X: x_data, Y: y_data})
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  형태로 출력
	
	# Accuracy report
	# hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, prediction, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```

이전 실행 결과 (2 Layers)
```
('\nHypothesis: ', array([[ 0.01338218], [ 0.98166394], [ 0.98809403], [ 0.01135799]], dtype=float32), 
 '\\Correct: '   , array([[ 0.],         [ 1.],         [ 1.],         [ 0.]]        , dtype=float32), 
 '\\Accuracy: '  , 1.0)
```

실행 결과 (4 Layers)
```
('\nHypothesis: ', array([[ 0.00159741], [ 0.99867946], [ 0.9982515 ], [ 0.0015059 ]], dtype=float32), 
 '\\Correct: '   , array([[ 0.],         [ 1.],         [ 1.],         [ 0.]]        , dtype=float32), 
 '\\Accuracy: '  , 1.0)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-3-xor-nn-wide-deep.py


<br /><br />







