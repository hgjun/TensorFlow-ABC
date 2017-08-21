<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 05. Logistic (Regression) Classification

Linear regression�� ���� �����ͷ� ���ο� ������ x �� ���� y�� �������� ���ڸ� ����   

Logistic classification�� binary classification  -> 
�� �� �ϳ��� �з��ϴ°�  




(��)
- Spam detection: �������� �ƴ���
- ���̽��� ��õ: ������ �ƴ���
- �ſ�ī�� �̻� ����: legitimate Ÿ�� / fraud


���� ��ũ  
https://www.youtube.com/watch?v=PIjno6paszY&feature=youtu.be  
http://gnujoow.github.io/ml/2016/01/29/ML3-Logistic-Regression/  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-2-logistic_regression_diabetes.py  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Binary classification

Binary classification�� ���� Logistic regression�� �ϰ�  
Logistic regression�� ���� Logistic function (sigmoid function)�� ����

�ֳ��ϸ�  
Binary classification�� ���� Linear regression ����ϸ� ����� ������ �ֱ� ����


1. ������ 1  

```
(��) x ���νð�, y �����հݿ���
x y
1 0
2 0
3 0
5 1
6 1
7 1
```

h(x) = wx = ( 0.5 * 1/4 ) w �� �н��Ǿ��ٸ� (x=4 �϶� y=0.5 �ǰ� ����)

�׷� ���� hypothesis�δ�  
x = 1,2,3 �̸� 0.5 ���ϴϱ� false ����  
y = 4,5,6 �̸� 0.5 �̻��̴� true ����  

�׷��� training data��   
x = 100 �̻��� ��,  y = 1 �� �ִٸ�  
���� ����� �ϸ����� �� (0.5 * 1/4) ���� ������  
�׷��� ���� x= 7�϶� y=0.5 �ǰ� ������  
������ x=5,6 �� �� y=1 true ���;� �ϳ�  
0.5 ���� ������ �Ǽ� false ��� �߸��� ������ �ϰ� ��  

2. ������ 2  

```
(��) h(x) = 0.5x ��
 x = 100 �� �� y = 50 (1���� �ſ� ū�� ����)
```

binary classification ���� y�� 0 or 1�� ���� ����� ����Ѵ�  
�׷��� h(x) =Wx + b �δ� linear�� ���̶�  
y�� 1���� �ſ� ū �� Ȥ�� 0���� �ſ� ���� ���� ���� �� �־�  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Logisitc Hypothesis

Logisitic function���� sigmoid function�� ����Ѵ�

sigmoid function ��(x) = 1 / (1 + e^(-x))

H(X) = 1 / (1 + e^(-��(tr(W)X + b)))

�׷��� y �� 0 ~ 1 ������ ���� ������ ��!


(cf)  
-��(tr(W)X + b) ����,  
b�� W�� w0�� �ְ� x1 ������ x0 ���� �����  
-��(tr(W)X) �ε� ǥ�� ����

���� ��ũ  
http://pythonkim.tistory.com/13


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost function

���� linear function�� cost function��
(mean squared error function) 2�� �Լ� ����   
�׷��� �̺��� �̿��� gradient descent ����� min error ����  

logistic function�� cost function���� mean squared error �� ����  
convex function�� ���� �ʾ� �ʹ� wavy�� �  
�̰� ����ϸ� local minima ���� ����!

�׷��� ������ ���� cost function ���  

Cost(W) = 1/m ��c( H(x), y ) 

c( H(x), y ): H(x), y�� ���� �Լ� c��,

if y = 1 then -log( H(x) )  
if y = 0 then -log( 1 - H(x) )

�켱 logisitic function�� �ڿ���� e �� ������ log ����ϱ� ����,  
-log(x) �� x = 1 �� �� y = 0,   x = 0 �϶� y �� ���Ѵ�� ���� ��̹Ƿ�  
���� y = 1 �� ��  h(x) = 1 �̸� error 0����, h(x) = 0 �̸� error �� �ִ�� ������ ��  

-log(1-x) �� x = 0 �� �� y = 0,   x = 1 �϶� y �� ���Ѵ�� ���� ��̹Ƿ�  
���� y = 0 �� ��  h(x) = 0 �̸� error 0����, h(x) = 1 �̸� error �� �ִ�� ������ ��


���� �� �� �� ���� ǥ���ϴ� ���� �� ���

c( H(x), y ) = - y log( H(x) ) - (1 - y) log( 1 - H(x) )

���� �̰� ����ϸ� gradient descent �����ϱ� ��������!

W <- W + ��W

W <- W - learning_rate * ��Cost(W) / ��W

�ټ��÷ο쿡���� ����ó�� ��� ����
```
cost = tf.reduce_sum( -Y * tf.log(hypo) - (1 - Y) * tf.log(1 - hypo)) / (m)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Logistic Classifier


H(X) = 1 / (1 + e^(-��(tr(W)X + b)))

Cost(W) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

W <- W - ��* ��Cost(W) / ��W


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
W = tf.Variable(tf.random_normal([2, 1]), name='weight') # X feature ����(2), Y ����(1) -> shape[2, 1] 2�� ���ͼ� 1�� ����
b = tf.Variable(tf.random_normal([1])   , name='bias'  ) # ������ �� (Y)�� ����

# hypothesis H(X) = 1 / (1 + e^(-��(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - ��* ��Cost(W) / ��W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis �� sigmoid function �̹Ƿ� 0 ~ 1 ������ ���̾�
# �̰� 0 or 1�� ���� Y ���� ���ϱ� ���ؼ��� hypo�� 0 or 1�� ��������
# �� h �� 0.5 ���� ũ�� 1 �ƴϸ� 0���� ��ȯ
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 �� �� 5�� true�� �������� accuracy = true���� / ��ü ���� = 0.5 �� ��.


# 2) Feed data and run graph (operation) using sess.run(op)
# sess = tf.Session()                         # launch the graph in a session
# sess.run(tf.global_variables_initializer()) # init global variables
# ������ python�� with ��� ����� ��

# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch the graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
		if step % 200 == 0:
			print(step, "Cost: ", cost_val)
			
	# Accuracy report: hypo, �����Ѱ�, accuracy(������ ����� ���� Y�� �� �� ����) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
	
	# ���� linear regression �ڵ�
	# for step in range(2001):
	#	cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={X: x_data, Y: y_data})
	#	if step % 10 == 0:
	#		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```

<br />
���� ��ũ  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Classifying diabetes

���� ���� ����غ��� (�索�� ������)

data02_diabetes.csv ������ ������  
C:\Users\(����� �����̸�)\Docker\work �� �����ؼ� ���

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
X = tf.placeholder(tf.float32, shape=[None, 8]) # X �Ӽ� ���� 8��
Y = tf.placeholder(tf.float32, shape=[None, 1]) # Y ���� 1��

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# hypothesis H(X) = 1 / (1 + e^(-��(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - ��* ��Cost(W) / ��W
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis �� sigmoid function �̹Ƿ� 0 ~ 1 ������ ���̾�
# �̰� 0 or 1�� ���� Y ���� ���ϱ� ���ؼ��� hypo�� 0 or 1�� ��������
# �� h �� 0.5 ���� ũ�� 1 �ƴϸ� 0���� ��ȯ
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 �� �� 5�� true�� �������� accuracy = true���� / ��ü ���� = 0.5 �� ��.


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
		if step % 200 == 0:
			print(step, "Cost: ", cost_val)
			
	# Accuracy report: hypo, �����Ѱ�, accuracy(������ ����� ���� Y�� �� �� ����) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
    # �� ������ �� 76% accuracy ����
	
	# ���� linear regression �ڵ�
	# for step in range(2001):
	#	cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={X: x_data, Y: y_data})
	#	if step % 10 == 0:
	#		print(step, "Cost: ", cost_val, "\bPrediction:\n", h_val)
```

<br />
���� ��ũ  
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
X = tf.placeholder(tf.float32, shape=[None, 8]) # X �Ӽ� ���� 8��
Y = tf.placeholder(tf.float32, shape=[None, 1]) # Y ���� 1��

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# hypothesis H(X) = 1 / (1 + e^(-��(tr(W)X)))
# h = tf.div(1., 1. + tf.exp(-(tf.matmul(X, W) + b)))
h = tf.sigmoid(tf.matmul(X, W) + b)

# cost function
# Cost(W) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))

# Minimize: small learning rate
# W <- W - ��* ��Cost(W) / ��W
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5 else False
# hypothesis �� sigmoid function �̹Ƿ� 0 ~ 1 ������ ���̾�
# �̰� 0 or 1�� ���� Y ���� ���ϱ� ���ؼ��� hypo�� 0 or 1�� ��������
# �� h �� 0.5 ���� ũ�� 1 �ƴϸ� 0���� ��ȯ
predicted = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
# 10 �� �� 5�� true�� �������� accuracy = true���� / ��ü ���� = 0.5 �� ��.


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

	# Accuracy report: hypo, �����Ѱ�, accuracy(������ ����� ���� Y�� �� �� ����) 
	hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Try other classification data

https://kaggle.com
