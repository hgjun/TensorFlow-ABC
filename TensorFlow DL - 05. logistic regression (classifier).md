<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 05. Logistic (Regression) Classification

Linear regression은 기존 데이터로 새로운 데이터 x 에 대해 y가 무엇일지 숫자를 예측   

Logistic classification은 binary classification  -> 
둘 중 하나로 분류하는것  




(예)
- Spam detection: 스팸일지 아닐지
- 페이스북 추천: 보일지 아닐지
- 신용카드 이상 결제: legitimate 타당 / fraud


참조 링크  
https://www.youtube.com/watch?v=PIjno6paszY&feature=youtu.be  
http://gnujoow.github.io/ml/2016/01/29/ML3-Logistic-Regression/  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-2-logistic_regression_diabetes.py  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Binary classification

Binary classification을 위해 Logistic regression을 하고  
Logistic regression을 위해 Logistic function (sigmoid function)을 쓴다

왜냐하면  
Binary classification을 위해 Linear regression 사용하면 생기는 문제들 있기 때문


1. 문제점 1  

```
(예) x 공부시간, y 시험합격여부
x y
1 0
2 0
3 0
5 1
6 1
7 1
```

h(x) = wx = ( 0.5 * 1/4 ) w 로 학습되었다면 (x=4 일때 y=0.5 되게 나옴)

그럼 구한 hypothesis로는  
x = 1,2,3 이면 0.5 이하니까 false 예측  
y = 4,5,6 이면 0.5 이상이니 true 예측  

그러나 training data에   
x = 100 이상의 값,  y = 1 이 있다면  
기울기 상당히 완만해질 것 (0.5 * 1/4) 보다 작은값  
그래서 만약 x= 7일때 y=0.5 되게 나오면  
실제는 x=5,6 일 때 y=1 true 나와야 하나  
0.5 보다 작은값 되서 false 라는 잘못된 예측을 하게 되  

2. 문제점 2  

```
(예) h(x) = 0.5x 면
 x = 100 일 때 y = 50 (1보다 매우 큰값 나와)
```

binary classification 에선 y는 0 or 1의 값을 만들어 줘야한다  
그러나 h(x) =Wx + b 로는 linear한 선이라  
y가 1보다 매우 큰 값 혹은 0보다 매우 작은 값을 가질 수 있어  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Logisitc Hypothesis

Logisitic function으로 sigmoid function을 사용한다

sigmoid function σ(x) = 1 / (1 + e^(-x))

H(X) = 1 / (1 + e^(-Σ(tr(W)X + b)))

그러면 y 는 0 ~ 1 사이의 값을 가지게 됨!


(cf)  
-Σ(tr(W)X + b) 에서,  
b를 W에 w0로 넣고 x1 이전에 x0 변수 만들면  
-Σ(tr(W)X) 로도 표현 가능

참조 링크  
http://pythonkim.tistory.com/13


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost function

예전 linear function의 cost function은
(mean squared error function) 2차 함수 형태   
그래서 미분을 이용한 gradient descent 사용해 min error 구함  

logistic function의 cost function으로 mean squared error 를 쓰면  
convex function이 되지 않아 너무 wavy한 곡선  
이걸 사용하면 local minima 문제 생김!

그래서 다음과 같은 cost function 사용  

Cost(W) = 1/m Σc( H(x), y ) 

c( H(x), y ): H(x), y에 대한 함수 c는,

if y = 1 then -log( H(x) )  
if y = 0 then -log( 1 - H(x) )

우선 logisitic function에 자연상수 e 가 있으니 log 사용하기 좋고,  
-log(x) 는 x = 1 일 때 y = 0,   x = 0 일때 y 가 무한대로 가는 곡선이므로  
실제 y = 1 일 때  h(x) = 1 이면 error 0으로, h(x) = 0 이면 error 값 최대로 나오게 함  

-log(1-x) 는 x = 0 일 때 y = 0,   x = 1 일때 y 가 무한대로 가는 곡선이므로  
실제 y = 0 일 때  h(x) = 0 이면 error 0으로, h(x) = 1 이면 error 값 최대로 나오게 함


따라서 위 두 개 같이 표현하는 다음 식 사용

c( H(x), y ) = - y log( H(x) ) - (1 - y) log( 1 - H(x) )

이제 이걸 사용하면 gradient descent 적용하기 수월해져!

W <- W + ΔW

W <- W - learning_rate * ∂Cost(W) / ∂W

텐서플로우에서는 다음처럼 사용 가능
```
cost = tf.reduce_sum( -Y * tf.log(hypo) - (1 - Y) * tf.log(1 - hypo)) / (m)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Logistic Classifier


H(X) = 1 / (1 + e^(-Σ(tr(W)X + b)))

Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

W <- W - η* ∂Cost(W) / ∂W


```
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
# Training data
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight') # X feature 개수(2), Y 개수(1) -> shape[2, 1] 2개 들어와서 1개 나와
b = tf.Variable(tf.random_normal([1])   , name='bias'  ) # 나가는 값 (Y)의 개수

# hypothesis H(X) = 1 / (1 + e^(-Σ(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis 는 sigmoid function 이므로 0 ~ 1 사이의 값이야
# 이걸 0 or 1인 실제 Y 값과 비교하기 위해서는 hypo도 0 or 1로 만들어줘야
# 즉 h 가 0.5 보다 크면 1 아니면 0으로 변환
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# sess = tf.Session()                         # launch the graph in a session
# sess.run(tf.global_variables_initializer()) # init global variables
# 앞으로 python의 with 기능 사용할 것

# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch the graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
		if step % 200 == 0:
			print(step, "Cost: ", cost_val)
			
	# Accuracy report: hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
	
	# 이전 linear regression 코드
	# for step in range(2001):
	#	cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={X: x_data, Y: y_data})
	#	if step % 10 == 0:
	#		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Classifying diabetes

실제 예제 사용해보기 (당뇨병 데이터)

data02_diabetes.csv 데이터 파일은  
C:\Users\(사용자 계정이름)\Docker\work 에 저장해서 사용

```
import tensorflow as tf
import numpy as np
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
xy = np.loadtxt('data02_diabetes.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 8]) # X 속성 개수 8개
Y = tf.placeholder(tf.float32, shape=[None, 1]) # Y 개수 1개

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# hypothesis H(X) = 1 / (1 + e^(-Σ(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis 는 sigmoid function 이므로 0 ~ 1 사이의 값이야
# 이걸 0 or 1인 실제 Y 값과 비교하기 위해서는 hypo도 0 or 1로 만들어줘야
# 즉 h 가 0.5 보다 크면 1 아니면 0으로 변환
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
		if step % 200 == 0:
			print(step, "Cost: ", cost_val)
			
	# Accuracy report: hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
    # 이 예제는 약 76% accuracy 보임
	
	# 이전 linear regression 코드
	# for step in range(2001):
	#	cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={X: x_data, Y: y_data})
	#	if step % 10 == 0:
	#		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-2-logistic_regression_diabetes.py

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Classifying diabetes with Queues

```
import tensorflow as tf
import numpy as np
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
filename_queue = tf.train.string_input_producer(
	['data02_diabetes.csv'], shuffle = False, name = 'filename_queue')

reader     = tf.TextLineReader()
key, value = reader.read(filename_queue)

# default values (for empty columns), type of decoded results
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)

# collect batches of csv
train_x_batch, train_y_batch = tf.train.batch(
	[xy[0:-1], xy[-1:]], batch_size = 10)

# old code
# xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype = np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 8]) # X 속성 개수 8개
Y = tf.placeholder(tf.float32, shape=[None, 1]) # Y 개수 1개

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# hypothesis H(X) = 1 / (1 + e^(-Σ(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis 는 sigmoid function 이므로 0 ~ 1 사이의 값이야
# 이걸 0 or 1인 실제 Y 값과 비교하기 위해서는 hypo도 0 or 1로 만들어줘야
# 즉 h 가 0.5 보다 크면 1 아니면 0으로 변환
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph
	sess.run(tf.global_variables_initializer())  # init global variables
	
	# Start populating the filename queue 
	coord   = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	
	for step in range(10001):
		x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
		
		cost_val, _ = sess.run([cost, train], feed_dict = {X: x_batch, Y: y_batch})
		if step % 200 == 0:
			print(step, "Cost: ", cost_val)
	
	coord.request_stop()
	coord.join(threads)

	# Accuracy report: hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Try other classification data

https://kaggle.com
