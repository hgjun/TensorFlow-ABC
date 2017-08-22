<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## Lab10-4. NN, ReLu, Xavier, Dropout, and Adam

NN 사용시 도움되는 팁들 살펴볼 것

(10장의 큰 주제: Deep learning 잘하는 방법들)


<br />
참조 링크  
https://www.youtube.com/watch?v=6CCXyfvubvY&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax Classifier for MNIST

(TensorFlow DL - 07. ml application & tips 2 with mnist data 참조)

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

# parameters
learning_rate   = 0.001
training_epochs = 15   # tr data 15번 반복해서 학습
batch_size      = 100  # 1 epoch (tr data 한번 학습) 시 100개씩 배치로 읽어
nb_classes      = 10   # 0 ~ 9 까지를 one-hot encoding 값으로 사용할 것

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
# TensorFlow ex - lab06. softmax classifier.txt  363 line 읽어
logits = tf.matmul(X, W) + b

# cost function: cross entropy cost/loss
# cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
# cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis = 1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test model and check accuracy
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), feed_dict={X: mnist.test.images[r:r + 1]})) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

실행 결과
```
('Epoch:', '0001', 'cost =', '6.316481819')
('Epoch:', '0002', 'cost =', '1.871113027')
('Epoch:', '0003', 'cost =', '1.215541895')
('Epoch:', '0004', 'cost =', '0.955135312')
('Epoch:', '0005', 'cost =', '0.810130810')
('Epoch:', '0006', 'cost =', '0.715441417')
('Epoch:', '0007', 'cost =', '0.646553366')
('Epoch:', '0008', 'cost =', '0.594896871')
('Epoch:', '0009', 'cost =', '0.554082964')
('Epoch:', '0010', 'cost =', '0.521780001')
('Epoch:', '0011', 'cost =', '0.494556942')
('Epoch:', '0012', 'cost =', '0.472523669')
('Epoch:', '0013', 'cost =', '0.453219114')
('Epoch:', '0014', 'cost =', '0.437310457')
('Epoch:', '0015', 'cost =', '0.422992098')
Learning finished
('Accuracy: ', 0.89590001)
('Label: ', array([7]))
('Prediction: ', array([7]))
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-1-mnist_softmax.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### NN for MNIST

3 layer에 ReLU 쓴 예제  
(3 layer 면 본격적인 deep nn 보단 그냥 softmax 에 가깝지만 그래도 실험해볼것)

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

# parameters
learning_rate   = 0.001
training_epochs = 15   # tr data 15번 반복해서 학습
batch_size      = 100  # 1 epoch (tr data 한번 학습) 시 100개씩 배치로 읽어

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

# weights and bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2, W3) + b3


# cost function: cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test model and check accuracy
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), feed_dict={X: mnist.test.images[r:r + 1]})) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

아까 1 layer NN 결과
```
('Epoch:', '0001', 'cost =', '6.316481819')
('Epoch:', '0002', 'cost =', '1.871113027')
('Epoch:', '0003', 'cost =', '1.215541895')
('Epoch:', '0004', 'cost =', '0.955135312')
('Epoch:', '0005', 'cost =', '0.810130810')
('Epoch:', '0006', 'cost =', '0.715441417')
('Epoch:', '0007', 'cost =', '0.646553366')
('Epoch:', '0008', 'cost =', '0.594896871')
('Epoch:', '0009', 'cost =', '0.554082964')
('Epoch:', '0010', 'cost =', '0.521780001')
('Epoch:', '0011', 'cost =', '0.494556942')
('Epoch:', '0012', 'cost =', '0.472523669')
('Epoch:', '0013', 'cost =', '0.453219114')
('Epoch:', '0014', 'cost =', '0.437310457')
('Epoch:', '0015', 'cost =', '0.422992098')
Learning finished
('Accuracy: ', 0.89590001)
('Label: ', array([7]))
('Prediction: ', array([7]))
```

지금 3 layer NN 결과 (더 좋음)
```
('Epoch:', '0001', 'cost =', '174.403094025')
('Epoch:', '0002', 'cost =', '41.436762782')
('Epoch:', '0003', 'cost =', '26.425457510')
('Epoch:', '0004', 'cost =', '18.364279145')
('Epoch:', '0005', 'cost =', '13.266407184')
('Epoch:', '0006', 'cost =', '9.945034577')
('Epoch:', '0007', 'cost =', '7.357624810')
('Epoch:', '0008', 'cost =', '5.598295127')
('Epoch:', '0009', 'cost =', '4.150287967')
('Epoch:', '0010', 'cost =', '3.140755168')
('Epoch:', '0011', 'cost =', '2.361504163')
('Epoch:', '0012', 'cost =', '1.820578709')
('Epoch:', '0013', 'cost =', '1.438388431')
('Epoch:', '0014', 'cost =', '1.088129262')
('Epoch:', '0015', 'cost =', '0.812648771')
Learning finished
('Accuracy: ', 0.9472)
('Label: ', array([5]))
('Prediction: ', array([5]))
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-2-mnist_nn.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Xavier Initialization for MNIST


```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

# parameters
learning_rate   = 0.001
training_epochs = 15   # tr data 15번 반복해서 학습
batch_size      = 100  # 1 epoch (tr data 한번 학습) 시 100개씩 배치로 읽어

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

# weights and bias for nn layers
# W1 = tf.Variable(tf.random_normal([784, 256])) 이전 코드

with tf.variable_scope("scope_layers"):
# 이거 있어야 다음 에러 안떠! https://www.tensorflow.org/programmers_guide/variable_scope
# ValueError: Variable W1 already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
	W1 = tf.get_variable("W1", shape = [784, 256], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([256]))
	L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

	W2 = tf.get_variable("W2", shape = [256, 256], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([256]))
	L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

	W3 = tf.get_variable("W3", shape = [256, 10], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([10]))
	logits = tf.matmul(L2, W3) + b3


# cost function: cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # 이거 있어야 Variable W1 already exists 에러 안떠
        # 처음 만들때는 reuse = False, 이후 만들어진 다음부터는 reuse = True 로
        if epoch > 1:
            tf.get_variable_scope().reuse_variables()
            
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test model and check accuracy
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), feed_dict={X: mnist.test.images[r:r + 1]})) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

이전 3 layer nn 결과
```
('Epoch:', '0001', 'cost =', '174.403094025') <-- 처음값 매우 커
('Epoch:', '0002', 'cost =', '41.436762782')
('Epoch:', '0003', 'cost =', '26.425457510')
('Epoch:', '0004', 'cost =', '18.364279145')
('Epoch:', '0005', 'cost =', '13.266407184')
('Epoch:', '0006', 'cost =', '9.945034577')
('Epoch:', '0007', 'cost =', '7.357624810')
('Epoch:', '0008', 'cost =', '5.598295127')
('Epoch:', '0009', 'cost =', '4.150287967')
('Epoch:', '0010', 'cost =', '3.140755168')
('Epoch:', '0011', 'cost =', '2.361504163')
('Epoch:', '0012', 'cost =', '1.820578709')
('Epoch:', '0013', 'cost =', '1.438388431')
('Epoch:', '0014', 'cost =', '1.088129262')
('Epoch:', '0015', 'cost =', '0.812648771')
Learning finished
('Accuracy: ', 0.9472)
('Label: ', array([5]))
('Prediction: ', array([5]))
```

3 layer nn with xavier init 결과 (더 좋음)
```
('Epoch:', '0001', 'cost =', '0.315164975')   <-- 처음부터 cost값 낮아 (initialize 잘됐다는 것)
('Epoch:', '0002', 'cost =', '0.114208306')
('Epoch:', '0003', 'cost =', '0.073980514')
('Epoch:', '0004', 'cost =', '0.053387901')
('Epoch:', '0005', 'cost =', '0.038962561')
('Epoch:', '0006', 'cost =', '0.029183927')
('Epoch:', '0007', 'cost =', '0.024885331')
('Epoch:', '0008', 'cost =', '0.018779066')
('Epoch:', '0009', 'cost =', '0.017799988')
('Epoch:', '0010', 'cost =', '0.014727625')
('Epoch:', '0011', 'cost =', '0.012383287')
('Epoch:', '0012', 'cost =', '0.008348369')
('Epoch:', '0013', 'cost =', '0.012307400')
('Epoch:', '0014', 'cost =', '0.010578755')
('Epoch:', '0015', 'cost =', '0.009228508')
Learning finished
('Accuracy: ', 0.97920001)
('Label: ', array([5]))
('Prediction: ', array([5]))
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-3-mnist_nn_xavier.py  
http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Deep NN for MNIST

더 deep, wide 하게 만들자

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

# parameters
learning_rate   = 0.001
training_epochs = 15   # tr data 15번 반복해서 학습
batch_size      = 100  # 1 epoch (tr data 한번 학습) 시 100개씩 배치로 읽어

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

# weights and bias for nn layers
with tf.variable_scope("scope_layers"):
	W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([512]))
	L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

	W2 = tf.get_variable("W2", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([512]))
	L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

	W3 = tf.get_variable("W3", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([512]))
	L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
	
	W4 = tf.get_variable("W4", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.Variable(tf.random_normal([512]))
	L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
	
	W5 = tf.get_variable("W5", shape = [512, 10], initializer = tf.contrib.layers.xavier_initializer())
	b5 = tf.Variable(tf.random_normal([10]))
	logits = tf.matmul(L4, W5) + b5
	
# cost function: cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # 이거 있어야 Variable W1 already exists 에러 안떠
        # 처음 만들때는 reuse = False, 이후 만들어진 다음부터는 reuse = True 로
        if epoch > 1:
            tf.get_variable_scope().reuse_variables()
            
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test model and check accuracy
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), feed_dict={X: mnist.test.images[r:r + 1]})) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

3 layer nn with xavier init 결과
```
('Epoch:', '0001', 'cost =', '0.315164975')
('Epoch:', '0002', 'cost =', '0.114208306')
('Epoch:', '0003', 'cost =', '0.073980514')
('Epoch:', '0004', 'cost =', '0.053387901')
('Epoch:', '0005', 'cost =', '0.038962561')
('Epoch:', '0006', 'cost =', '0.029183927')
('Epoch:', '0007', 'cost =', '0.024885331')
('Epoch:', '0008', 'cost =', '0.018779066')
('Epoch:', '0009', 'cost =', '0.017799988')
('Epoch:', '0010', 'cost =', '0.014727625')
('Epoch:', '0011', 'cost =', '0.012383287')
('Epoch:', '0012', 'cost =', '0.008348369')
('Epoch:', '0013', 'cost =', '0.012307400')
('Epoch:', '0014', 'cost =', '0.010578755')
('Epoch:', '0015', 'cost =', '0.009228508')
Learning finished
('Accuracy: ', 0.97920001)
('Label: ', array([5]))
('Prediction: ', array([5]))
```

deep nn 결과
```
('Epoch:', '0001', 'cost =', '0.297078801')
('Epoch:', '0002', 'cost =', '0.104386068')
('Epoch:', '0003', 'cost =', '0.070857233')
('Epoch:', '0004', 'cost =', '0.052712927')
('Epoch:', '0005', 'cost =', '0.041292441')
('Epoch:', '0006', 'cost =', '0.035878071')
('Epoch:', '0007', 'cost =', '0.027150498')
('Epoch:', '0008', 'cost =', '0.028746802')
('Epoch:', '0009', 'cost =', '0.024041419')
('Epoch:', '0010', 'cost =', '0.022092675')
('Epoch:', '0011', 'cost =', '0.018696169')
('Epoch:', '0012', 'cost =', '0.015694544')
('Epoch:', '0013', 'cost =', '0.017981618')
('Epoch:', '0014', 'cost =', '0.014740398')
('Epoch:', '0015', 'cost =', '0.014290565')
Learning finished
('Accuracy: ', 0.97460002)
('Label: ', array([6]))
('Prediction: ', array([6]))
```
<br />

그런데 아까보다 accuracy 좀 안좋아졌어 overfitting 때문일 것  
==> Dropout 도 쓰자

참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-4-mnist_nn_deep.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Dropout for MNIST

dropout 레이어 하나 더 추가

.. - [L1] - [dropout L1] - ...

(cf)

tf 1.0 부터는 arg 이름 keep_prob 사용  
```
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)
```

전체 코드
```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

# parameters
learning_rate   = 0.001
training_epochs = 15   # tr data 15번 반복해서 학습
batch_size      = 100  # 1 epoch (tr data 한번 학습) 시 100개씩 배치로 읽어

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate 0.7 (training) and  1 (testing)
keep_prob = tf.placeholder(tf.float32)

# weights and bias for nn layers
with tf.variable_scope("scope_layers"):
	W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([512]))
	L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
	L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

	W2 = tf.get_variable("W2", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([512]))
	L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
	L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

	W3 = tf.get_variable("W3", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([512]))
	L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
	L3 = tf.nn.dropout(L3, keep_prob = keep_prob)
	
	W4 = tf.get_variable("W4", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.Variable(tf.random_normal([512]))
	L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
	L4 = tf.nn.dropout(L4, keep_prob = keep_prob)
	
	W5 = tf.get_variable("W5", shape = [512, 10], initializer = tf.contrib.layers.xavier_initializer())
	b5 = tf.Variable(tf.random_normal([10]))
	logits = tf.matmul(L4, W5) + b5
	
# cost function: cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # 이거 있어야 Variable W1 already exists 에러 안떠
        # 처음 만들때는 reuse = False, 이후 만들어진 다음부터는 reuse = True 로
        if epoch > 1:
            tf.get_variable_scope().reuse_variables()
            
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7} # dropout rate 0.7 사용
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test model and check accuracy
    # 테스트할 때는 반드시 dropout rate = 1 로!
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1})) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

이전 deep nn 결과
```
('Epoch:', '0001', 'cost =', '0.297078801')
('Epoch:', '0002', 'cost =', '0.104386068')
('Epoch:', '0003', 'cost =', '0.070857233')
('Epoch:', '0004', 'cost =', '0.052712927')
('Epoch:', '0005', 'cost =', '0.041292441')
('Epoch:', '0006', 'cost =', '0.035878071')
('Epoch:', '0007', 'cost =', '0.027150498')
('Epoch:', '0008', 'cost =', '0.028746802')
('Epoch:', '0009', 'cost =', '0.024041419')
('Epoch:', '0010', 'cost =', '0.022092675')
('Epoch:', '0011', 'cost =', '0.018696169')
('Epoch:', '0012', 'cost =', '0.015694544')
('Epoch:', '0013', 'cost =', '0.017981618')
('Epoch:', '0014', 'cost =', '0.014740398')
('Epoch:', '0015', 'cost =', '0.014290565')
Learning finished
('Accuracy: ', 0.97460002)
('Label: ', array([6]))
('Prediction: ', array([6]))
```

deep nn with dropout 결과 (98% !!)
```
('Epoch:', '0001', 'cost =', '0.450323299')
('Epoch:', '0002', 'cost =', '0.173080627')
('Epoch:', '0003', 'cost =', '0.128456958')
('Epoch:', '0004', 'cost =', '0.109852645')
('Epoch:', '0005', 'cost =', '0.093579261')
('Epoch:', '0006', 'cost =', '0.083109432')
('Epoch:', '0007', 'cost =', '0.079094050')
('Epoch:', '0008', 'cost =', '0.070236224')
('Epoch:', '0009', 'cost =', '0.062607277')
('Epoch:', '0010', 'cost =', '0.056392715')
('Epoch:', '0011', 'cost =', '0.058960581')
('Epoch:', '0012', 'cost =', '0.053105334')
('Epoch:', '0013', 'cost =', '0.050874679')
('Epoch:', '0014', 'cost =', '0.048412906')
('Epoch:', '0015', 'cost =', '0.044575504')
Learning finished
('Accuracy: ', 0.98220003)
('Label: ', array([5]))
('Prediction: ', array([5]))
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-5-mnist_nn_dropout.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Optimizers

지금까지 이거 주로 썼어
```
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```

Optimizer도 여러개 있어
- tf.train.AdadeltaOptimizer
- tf.train.AdagradOptimizer
- tf.train.AdagradDAOptimizer
- tf.train.MomentumOptimizer
- tf.train.AdamOptimizer
- tf.train.FtrlOptimizer
- tf.train.ProximalGradientDescentOptimizer
- tf.train.ProximalAdagradOptimizer
- tf.train.RMSPropOptimizer

Optimizer 비교한 그림  
http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html 

ADAM optimizer 가 좋은 결과 만든다는 논문  
ADAM: a method for stochastic optimization [Kingma et al. 2015]


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### ADAM Optimizer

ADAM Optimizer 권장!

ADAM 부터 쓰고 잘 안되면 다른거 써보는 식으로 하면 됨

```
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                 logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Summary

- Softmax vs. Neural Nets for MNIST
  - 90% vs. 94.5%

- Xavier initialization
  - 97.8%

- Deep Neural Nets with Dropout
  - 98%

- Adam optimizer 로 시작하고 잘 안되면 other optimizers 사용



