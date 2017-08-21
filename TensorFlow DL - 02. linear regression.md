<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 02. Linear Regression

H(x) = Wx + b

매트릭스 사용한 수식은 X 먼저 써 

H(X) = XW + b

cost(W, b) = min squared error

참조 링크  
https://www.youtube.com/watch?v=mQGwjrStQgg&feature=youtu.be  
https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80  
http://pythonkim.tistory.com/13


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Linear Regression

```
import tensorflow as tf
# 1) Build a graph using tf operations
# H(x) = Wx + b

# training data X, Y
# placeholder 사용 가능
x_tr = [1, 2, 3]
y_tr = [1, 2, 3]

# variable 
# 기존 변수와 다른 개념임 (trainable variable)
# 트레이닝을 통해 학습되야할 파라메터
# 트레이닝의 결과가 담길 곳으로 tf 의해 계속 트레이닝(modified)되는 노드
# variable 은 초기화해서 사용해야 함 (tf.global_variables_initializer())
# placeholder는 일종의 input, variable은 중간 과정에 필요한 변수

# 값 아직 모르니 랜덤한 값으로 줌 shape 는 크기 1인 벡터로 ([1])
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis: XW + b
h = x_tr * W + b

# cost(loss) function
# t = [1., 2., 3., 4.]
# print(sess.run(tf.reduce_mean(t)))
# reduce_mean 결과는 평균: 2.5
m = len(x_tr)
cost = tf.reduce_sum(tf.pow(h - Y, 2))/(m)
# cost = tf.reduce_mean(tf.square(h - y_tr)) 한 줄에 가능


# Min cost: GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)
# fit the line
for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b)) # 각 노드 cost, W, b 값 출력


# Graph 예제
(train) - (cost) - (y_tr)
                 - (h)    -  (x_tr * W + b)
```

참조 링크  
http://jrmeyer.github.io/misc/tf-graph.png  
http://3months.tistory.com/68  
http://www.popit.kr/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0tensorflow-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0/  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Linear Regression: Placehloders 사용

```
import tensorflow as tf
# 1) Build a graph using tf operations

# training data X, Y
# tensor will be fed using feed_dict
# http://stackoverflow.com/questions/36693740
# shape=[None] 아무 형태나 입력 가능하다
X = tf.placeholder(tf.float32, shape=[None])    # x_tr = [1, 2, 3]
Y = tf.placeholder(tf.float32, shape=[None])    # y_tr = [1, 2, 3]


# variables
# tf.random_normal([1], seed=1) 로 난수 일정한 값 계속 사용 가능
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis: XW + b
h = X * W + b

# cost(loss) function
cost = tf.reduce_mean(tf.square(h - Y))

# Min cost: GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)

# placehode 로 학습데이터 x, y를 원하는대로 입력해줄 수 있다 
# 옛날 예제 print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))

# fit the line
for step in range(2001):
	# run 여러개 한번에 실행 가능
	# train 으로 나온 value는 필요없어서 _ 로 생략
	cost_val, W_val, b_val, _ = \
		sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
		# W = 1, b = 1.1 로 수렴될 것
	if step % 20 == 0:
		print(step, cost_val, W_val, b_val) # 각 노드 cost, W, b 값 출력

# Testing the model
print(sess.run(h, feed_dict={X: [5]}))
print(sess.run(h, feed_dict={X: [2.5]}))
print(sess.run(h, feed_dict={X: [1.5, 3.5]}))
# [ 6.10049725]
# [ 3.5996027]
# [ 2.59924507  4.5999608 ]
```


