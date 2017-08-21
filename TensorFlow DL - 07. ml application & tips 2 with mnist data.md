<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 07. Application & Tips 2

1. Training / Test datasets
2. Learning Rate
3. Input Normalization
4. Training and Evaluation (with MNIST data)


참조 링크
https://www.youtube.com/watch?v=oSJfejG2C3w&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training and Test datasets


```
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# training data
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy   = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

	# 학습: tr data 사용
    for step in range(201):
        cost_val, W_val, _ = sess.run(
            [cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

	# 테스트: test data 사용
    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
```

<br />
참조 링크
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-1-learning_rate_and_evaluation.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 2. Learning rate

- Learning rate
  - 너무 크면 overshooting 수렴해야하는데 튕겨나가 발산
  - 너무 작으면 iteration 많아져 느려, local minima 생길수


Learning rate 너무 큰 경우,  
위 예제 코드에서 learning_rate = 10 적용
```
learning_rate = 10
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
```

NaN 값 생겨 학습 포기
```
(0, 8.1185646, array([[  2.77571678,   3.12156439,  -5.76682711],
       [ 15.4650259 ,  18.74593163, -31.18079185],
       [ 16.88646698,  18.35827827, -34.85777283]], dtype=float32))
(1, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(2, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
...
```

Learning rate 너무 작은 경우,  
위 예제 코드에서 learning_rate = 1e-10 적용
```
learning_rate = 1e-10 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
```

값 너무 작게 변하는거 볼수 
```
(0, 8.3721457, array([[-0.16361724, -0.6074751 ,  0.28116855],
       [-0.06498256,  1.16052759,  0.37132689],
       [-0.66239232,  1.42011869, -1.24043822]], dtype=float32))
(1, 8.3721457, array([[-0.16361724, -0.6074751 ,  0.28116855],
       [-0.06498256,  1.16052759,  0.37132689],
       [-0.66239232,  1.42011869, -1.24043822]], dtype=float32))
(2, 8.3721457, array([[-0.16361724, -0.6074751 ,  0.28116855],
       [-0.06498256,  1.16052759,  0.37132689],
       [-0.66239232,  1.42011869, -1.24043822]], dtype=float32))
...
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 3. Input Normalization

#### Non-normalized inputs

data가 normalized 되어있지 않으면 learning rate 잘 설정해도 NaN 나올 수 있어

```
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# 큰 차이 있는 값들 있어 -> 학습 하다 NaN 발생시킴
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Normalized inputs (min-max scale)


MinMaxScaler() 함수 요새 많이 사용  
사용한 후 아까 xy 값 다음과 같아져 (0 ~ 1 사이 값으로 normalize)

```
[[ 0.99999999  0.99999999  0.          1.          1.        ]
 [ 0.70548491  0.70439552  1.          0.71881782  0.83755791]
 [ 0.54412549  0.50274824  0.57608696  0.606468    0.6606331 ]
 [ 0.33890353  0.31368023  0.10869565  0.45989134  0.43800918]
 [ 0.51436     0.42582389  0.30434783  0.58504805  0.42624401]
 [ 0.49556179  0.42582389  0.31521739  0.48131134  0.49276137]
 [ 0.11436064  0.          0.20652174  0.22007776  0.18597238]
 [ 0.          0.07747099  0.5326087   0.          0.        ]]
```

이제 learning_rate=1e-5 로 작아도 잘 학습된다.


```
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# very important. It does not work without it.
xy = MinMaxScaler(xy)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-2-linear_regression_without_min_max.py  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-3-linear_regression_min_max.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 4. Training and Evaluation with MNIST data

#### MNIST data

실제 데이터로 모델 만들어보기

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

#/home/testu/work/MNIST_data 에 다운 받음
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   
(처음 읽을 때 웹에서 다운 받으므로 시간 조금 걸림)



읽어올 때 바로 one-hot 으로 처리
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

이미지 하나 28 x 28 x 1 image = 총 784 픽셀
X = tf.placeholder(tf.float32, [None, 784])

Y 0 ~ 9의 숫자 이걸 one-hot 인코딩해서 10 class 가 될 것
Y = tf.placeholder(tf.float32, [None, 10]) 

메모리에 데이터 다 올리면 NG
배치로 (예, 100개씩 올려서) 사용
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-4-mnist_introduction.py
http://yann.lecun.com/exdb/mnist


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Training epoch/batch


##### 1 epoch  
1 forward pass and 1 backward pass of all the training examples  
전체 tr data 다 학습시키면 1 epoch


##### batch size  
the number of training examples in one forward/backward pass.  
The higher the batch size, the more memory space you'll need.  
1 epoch 위해 한번에 tr data 다 읽지 않고 100개씩 잘라서 씀 100 이 batch size
             
##### number of iterations  
number of passes, each pass using [batch size] number of examples.  
To be clear, one pass = one forward pass + one backward pass   
(we do not count the forward pass and backward pass as two different passes).

##### example
if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.  
1000개 tr data 이고 batch size 500 이면 1 epoch (전체 학습) 위해 2 iteration 반복됨


(코드 예)
```
training_epochs = 15   # 15 epoch 학습할 것 (많을수록 좋지만 느려지겠지)
batch_size = 100
for epoch in range(training_epochs):   # 15번 반복 
	avg_cost = 0
	
	# (예) 전체 데이터(1000) / 배치 사이즈 (500) = 2번 돌아
	total_batch = int(mnist.train.num_examples / batch_size)

	for i in range(total_batch): # (예) 1 epoch 위해 2번 돌아
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		c, _ = sess.run([cost, optimizer], feed_dict={
						X: batch_xs, Y: batch_ys})
		avg_cost += c / total_batch

	print('Epoch:', '%04d' % (epoch + 1),
		  'cost =', '{:.9f}'.format(avg_cost))
```

<br />
참조 링크  
https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Report results on test dataset

테스트 데이터로 테스트 할 것
```
print("Accuracy: ", accuracy.eval(session=sess, 
                    feed_dict= {X: mnist.test.images, Y: mnist.test.labels}))
```

이전에는 sess.run 사용했어
```
sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels})
```

이거 대신 accuracy 변수 직접 사용해서 accuracy.eval( ) 로도 가능
```
sess.run(var, ... )  <==>  var.eval( session = sess, ..)  같다!
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Sample image show and prediction



랜덤하게 숫자 이미지 하나 읽어와  
mnist.test.num_examples = 10000 이면,  
random.randint(0, 9999) 중 한 숫자 뽑아줄 것

```
import matplotlib.pyplot as plt
...

print("number of ex:", mnist.test.num_examples) # 10000
r = random.randint(0, mnist.test.num_examples - 1)
print(r)   # 4752 면 이미지 "9" 임


# one-hot 값이니 숫자 9 이미지 읽혔으면 argmax 의해 label = 9 될 것
print(mnist.test.labels[r:r + 1]) # [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))

# hypo 에 읽어온 이미지 주고 fitting 결과에서 argmax 출력
print("Prediction: ", sess.run(
	tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
```

(예) 학습 잘 됐으면 다음처럼 나올 것
```
('Label: ', array([9]))
('Prediction: ', array([9]))
```

읽어온 이미지 출력해줌
```
plt.imshow(
	 mnist.test.images[r:r + 1].reshape(28, 28),
	 cmap='Greys',
	 interpolation='nearest')
plt.show()
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Softmax classificaiton with MNIST data

전체 코드

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data #/home/testu/work/MNIST_data 에 다운 받음
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # cross entropy 사용한 cost함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
    
```

결과
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
('Epoch:', '0001', 'cost =', '2.539776730')
('Epoch:', '0002', 'cost =', '1.062999175')
('Epoch:', '0003', 'cost =', '0.856880818')
('Epoch:', '0004', 'cost =', '0.752523987')
('Epoch:', '0005', 'cost =', '0.686657698')
('Epoch:', '0006', 'cost =', '0.639839376')
('Epoch:', '0007', 'cost =', '0.604043013')
('Epoch:', '0008', 'cost =', '0.575251328')
('Epoch:', '0009', 'cost =', '0.552010339')
('Epoch:', '0010', 'cost =', '0.532184768')
('Epoch:', '0011', 'cost =', '0.515180909')
('Epoch:', '0012', 'cost =', '0.500344568')
('Epoch:', '0013', 'cost =', '0.487508247')
('Epoch:', '0014', 'cost =', '0.476209609')
('Epoch:', '0015', 'cost =', '0.465052148')
Learning finished
('Accuracy: ', 0.88959998)
('Label: ', array([7]))
('Prediction: ', array([7]))
```

