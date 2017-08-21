<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 09-3. TensorBoard for NN


#### TensorBoard

TensorFlow logging/debugging tool

예전에 실험 결과 보려면 print() 사용했어  
-> TensorBoard 로 GUI 지원


<br />
참조 링크  
https://www.youtube.com/watch?v=lmrWZPFYjHM&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 5 Steps of Using TensorBoard

#### 1. From TF graph, decide which tensors you want to log

로그로 남겨서 볼 데이터 선택
```
w2_hist  = tf.summary.histogram("weight2", W2) # 여러값일 때 -> histogram
cost_sum = tf.summary.scalar("cost", cost)     # 하나일 때   -> scalar
```

#### 2. Merge all summaries
```
summary = tf.summary.merge_all()
```

#### 3. Create writer and add graph
```
writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)
```
#### 4. Run merged summary merge and write add_summary

summary 도 tensor니 실행시켜야
```
s, _ = sess.run([summary, optimizer], feed_dict = feed_dict)
writer.add_summary(s, global_step = global_step) 그래프 그려줘
```

####5. Launch TensorBoard
```
tensorboard --logdir=./logs
기본 port 6006
```

remote server 쓰는 경우 포트포워딩 사용
```
ssh -L local_port:127.0.0.1:remort_port usernamer@server.com
```

로컬 터미널에서
```
ssh -L local_port:127.0.0.1:6006 myserver@server.com
```

서버 터미널에서
```
tensorboard --logdir=./logs/xor_logs
```

웹브라우저에서  
127.0.0.1:6006 로 사용 가능


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Graph Hierarchy

scope 로 graph hierarchy 정리 가능

```
with tf.name_scope("layer1") as scope:
	W1 =
	b1 =
	
with tf.name_scope("layer2") as scope:
	W2 =
	b2 =
	
...
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multiple Runs

learning_rate = 0.1 vs. learning_rate = 0.01  비교 가능

```
tensorboard -logdir=./logs/xor_logs_run_001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)
..
writer = tf.summary.FileWriter("/logs/xor_logs_run_001")


tensorboard -logdir=./logs/xor_logs_run_002
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)
..
writer = tf.summary.FileWriter("/logs/xor_logs_run_002")
```

터미널에서 상위 디렉토리로 실행  
tensorboard --logdir=./logs  
그래프에서 다 보여줘  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### XOR NN with TensorBoard

```
import tensorflow as tf
import numpy as np
import os
os.chdir('/home/testu/work') 
os.getcwd()

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
with tf.name_scope("layer1") as scope:
	W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')  # X feature 개수(2), Layer 개수(2) -> shape[2, 2] 2개 들어와서 2개 나와
	b1 = tf.Variable(tf.random_normal([2])   , name='bias1'  )  # Layer 2 개수(2)
	layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

	w1_hist = tf.summary.histogram("weights1", W1)
	b1_hist = tf.summary.histogram("biases1", b1)
	layer1_hist = tf.summary.histogram("layer1", layer1)
	
# Layer 2
with tf.name_scope("layer2") as scope:
	W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')  # 입력 개수(2), Y 개수 (2) -> shape[2, 1] 2개 들어와서 1개 나와
	b2 = tf.Variable(tf.random_normal([1])   , name='bias2'  )  # Y 개수 (2)
	h = tf.sigmoid(tf.matmul(layer1, W2) + b2)

	w2_hist = tf.summary.histogram("weights2", W2)
	b2_hist = tf.summary.histogram("biases2", b2)
	
# cost function
# Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )
with tf.name_scope("cost") as scope:
	cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))
	
	cost_summ = tf.summary.scalar("cost", cost)


# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
with tf.name_scope("train") as scope:
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

	# tensorboard --logdir=./logs/xor_logs
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter("./logs/xor_logs_run_001")
	writer.add_graph(sess.graph)
	
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(10001):
		summary, _ = sess.run([merged_summary, train], feed_dict = {X: x_data, Y: y_data})
		writer.add_summary(summary, global_step = step)
		
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  형태로 출력
	
	# Accuracy report
	# hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	hy, c, a = sess.run([h, prediction, accuracy], feed_dict = {X: x_data, Y: y_data})
	print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```

<br />
TensorFlow 컨테이너 (tftest) 계정의 터미널에서
```
tensorboard --logdir=/home/testu/work/logs
```
실행하고

웹 브라우저  
http://192.168.99.100:6006  
들어가면 TensorBoard 볼 수 있음

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-4-xor_tensorboard.py


<br /><br />
