<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 03. Minimizing Cost

참조 링크  
https://www.youtube.com/watch?v=Y0EF9VqRuEA&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost function plot

```
import tensorflow as tf
import matplotlib.pyplot as plt

# 1) Build a graph using tf operations
# training data X, Y
X = [1, 2, 3]
Y = [1, 2, 3]

# weight
W = tf.placeholder(tf.float32)

# simplyfied hypothesis: H(x) = XW   without bias
h = X * W

# cost(loss) function
cost = tf.reduce_mean(tf.square(h - Y))


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)

plot_x = []  # x축 weight W (-3 ~ 5)
plot_y = []  # y축 cost

for i in range(-30, 50):
	feed_W = i * 0.1
	curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
	plot_x.append(curr_W)
	plot_y.append(curr_cost)

# plot cost function
plt.plot(plot_x, plot_y)
plt.show()
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Gradient descent

- hypothesis h
  - h = X * W

- cost function (error function)
  - E[W] = tf.reduce_mean(tf.square(h - Y))

- gradient: ΔE[W] 
  - 경사, cost function의 기울기값
  - Δ (대문자 델타) 미분기호
  - η (소문자 이타) 학습률, 충분히 작으면 min error 계산 항상 converge
  - ΔW = - ηΔE[W] = η  Σ (h - Y)X = η  Σ (X * W - Y)X

- gradient descent
  - W = W + ΔW = W - ηΔE[W]
  - ηΔE[W]: - 기호 때문에 기울기 음방향이면 W 늘려줌, 양방향이면 줄여줌

<br />
```
import tensorflow as tf

# 1) Build a graph using tf operations
# training data X, Y
x_data = [1, 2, 3]
y_data = [1, 2, 3]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# variables
W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# simplyfied hypothesis: H(x) = XW   without bias
h = X * W

# cost(loss) function
cost = tf.reduce_mean(tf.square(h - Y))

# Min cost: Gradient Descent 이거 안쓰고 직접 계산해서 써볼 것
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# Gradient Descent using derivateive: W = W - learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y) * X)
descent  = W - learning_rate * gradient
update   = W.assign(descent)


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)

for step in range(21):
	sess.run(update, feed_dict={X: x_data, Y: y_data})
	cost_val, W_val = sess.run([cost, W], feed_dict={X: x_data, Y: y_data})
	print(step, cost_val, W_val)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Gradient descent ex2

```
import tensorflow as tf

# 1) Build a graph using tf operations
# training data X, Y
X = [1, 2, 3]
Y = [1, 2, 3]

# variables
# W = tf.Variable(tf.random_normal([1]), name='weight')
W = tf.Variable(5.0)
# W = tf.Variable(-30.0)

# simplyfied hypothesis: H(x) = XW   without bias
h = X * W

# cost(loss) function
cost = tf.reduce_mean(tf.square(h - Y))

# Min cost: tf 제공 operation 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)


for step in range(30):
	print(step, sess.run(W))
	sess.run(train)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Gradient descent: compute_gradient, apply_gradient

```
import tensorflow as tf

# 1) Build a graph using tf operations
# training data X, Y
X = [1, 2, 3]
Y = [1, 2, 3]

# variables
# W = tf.Variable(tf.random_normal([1], seed=1), name='weight')
W = tf.Variable(5.0)
# W = tf.Variable(-30.0)

# simplyfied hypothesis: H(x) = XW   without bias
h = X * W

# cost(loss) function
cost = tf.reduce_mean(tf.square(h - Y))

# Min cost: 내가 gradient 좀 수정해서 사용가능
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost) 바로 minimize 안하고
gvs = optimizer.compute_gradients(cost, [W])      # gradients 값 받아서 필요한 부분 수정
apply_gradients =  optimizer.apply_gradients(gvs) # gradients 수정한 값으로 적용


# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# 3) Update variables in the graph (and return values)

# manual gradient
manual_gradient = tf.reduce_mean((W*X - Y) * X) * 2

for step in range(30):
	# print(step, sess.run(W))
	# sess.run(train)
	print(step, sess.run([manual_gradient, W, gvs]))
	# 아무 수정안하면 manual gradient와 tf 제공op 로 계산한 gradient와 동일함 
	sess.run(apply_gradients)
```

