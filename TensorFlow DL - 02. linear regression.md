<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 02. Linear Regression

H(x) = Wx + b

��Ʈ���� ����� ������ X ���� �� 

H(X) = XW + b

cost(W, b) = min squared error

���� ��ũ  
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
# placeholder ��� ����
x_tr = [1, 2, 3]
y_tr = [1, 2, 3]

# variable 
# ���� ������ �ٸ� ������ (trainable variable)
# Ʈ���̴��� ���� �н��Ǿ��� �Ķ����
# Ʈ���̴��� ����� ��� ������ tf ���� ��� Ʈ���̴�(modified)�Ǵ� ���
# variable �� �ʱ�ȭ�ؼ� ����ؾ� �� (tf.global_variables_initializer())
# placeholder�� ������ input, variable�� �߰� ������ �ʿ��� ����

# �� ���� �𸣴� ������ ������ �� shape �� ũ�� 1�� ���ͷ� ([1])
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis: XW + b
h = x_tr * W + b

# cost(loss) function
# t = [1., 2., 3., 4.]
# print(sess.run(tf.reduce_mean(t)))
# reduce_mean ����� ���: 2.5
m = len(x_tr)
cost = tf.reduce_sum(tf.pow(h - Y, 2))/(m)
# cost = tf.reduce_mean(tf.square(h - y_tr)) �� �ٿ� ����


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
		print(step, sess.run(cost), sess.run(W), sess.run(b)) # �� ��� cost, W, b �� ���


# Graph ����
(train) - (cost) - (y_tr)
                 - (h)    -  (x_tr * W + b)
```

���� ��ũ  
http://jrmeyer.github.io/misc/tf-graph.png  
http://3months.tistory.com/68  
http://www.popit.kr/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0tensorflow-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0/  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Linear Regression: Placehloders ���

```
import tensorflow as tf
# 1) Build a graph using tf operations

# training data X, Y
# tensor will be fed using feed_dict
# http://stackoverflow.com/questions/36693740
# shape=[None] �ƹ� ���³� �Է� �����ϴ�
X = tf.placeholder(tf.float32, shape=[None])    # x_tr = [1, 2, 3]
Y = tf.placeholder(tf.float32, shape=[None])    # y_tr = [1, 2, 3]


# variables
# tf.random_normal([1], seed=1) �� ���� ������ �� ��� ��� ����
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

# placehode �� �н������� x, y�� ���ϴ´�� �Է����� �� �ִ� 
# ���� ���� print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))

# fit the line
for step in range(2001):
	# run ������ �ѹ��� ���� ����
	# train ���� ���� value�� �ʿ��� _ �� ����
	cost_val, W_val, b_val, _ = \
		sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
		# W = 1, b = 1.1 �� ���ŵ� ��
	if step % 20 == 0:
		print(step, cost_val, W_val, b_val) # �� ��� cost, W, b �� ���

# Testing the model
print(sess.run(h, feed_dict={X: [5]}))
print(sess.run(h, feed_dict={X: [2.5]}))
print(sess.run(h, feed_dict={X: [1.5, 3.5]}))
# [ 6.10049725]
# [ 3.5996027]
# [ 2.59924507  4.5999608 ]
```


