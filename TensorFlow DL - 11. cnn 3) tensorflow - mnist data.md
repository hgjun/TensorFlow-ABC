<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 11-3. CNN Tensorflow with MNIST data


11-2 내용 바탕으로 MNIST 분석하는 CNN 만들어볼 것 (Accuracy 99%)

<br />
참조 링크

https://www.youtube.com/watch?v=pQ9Y9ZagZBk&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1. Simple CNN

```
      input layer
           ↓
   Convolutional layer1
           ↓
     Pooling layer1
           ↓
   Convolutional layer2 
           ↓
     Pooling layer2
           ↓
   Fully Connected layer
           ↓
      Output layer
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 1.1 Conv Layer1

```
# MNIST data image of shape 28 * 28 = 784
# None 은 몇개 instance 들어올지 명시 안하는 것. n개의 이미지 들어올 것이다
X = tf.placeholder(tf.float32, [None, 784])

# 입력 형식에 맞게 reshape
# n개의 데이터 (-1), h x w 크기 (28 x 28), 1개 색깔
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28 x 28 x 1 (black/white)

# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

# 필터 사용해서 convolution 돌릴 것
# 필터 크기 (3 x 3 x 1), 32개 필터 사용
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))


# 필터는 한칸씩 움직이고 stride = (1 x 1)
# 입력 이미지와 같은 사이즈의 출력이미지를 만들어내 padding = SAME
# 단 32개 필터 사용하니 32개 이미지 생성됨 => (?, 28, 28, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print(L1)

# ReLU 통과시킴 (출력 사이즈 변화는 없음)
L1 = tf.nn.relu(L1)
print(L1)

# Max Pooling
# 슬라이딩 윈도우 크기 (2 x 2)  ksize = [1, 2, 2, 1]
# 두칸씩 움직여 stride = (2 x 2)
# zero padding 사용하므로 padding = SAME
# 출력 결과는 입력 받은 이미지의 반 => (?, 14, 14, 32)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
print(L1)

'''
print(L1)
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 1.2 Conv Layer2

```
# 입력 이미지 (conv layer1 로부터 받은) shape 는 (?, 14 x 14 x 32)
# 필터 사용해서 convolution 돌릴 것
# 필터 크기 (3 x 3 x 32), 64개 필터 사용
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))


# 필터는 한칸씩 움직이고 stride = (1 x 1)
# 입력 이미지와 같은 사이즈의 출력이미지를 만들어내 padding = SAME
# 단 64개 필터 사용하니 64개 이미지 생성됨 => (?, 14, 14, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print(L2)

# ReLU 통과시킴 (출력 사이즈 변화는 없음)
L2 = tf.nn.relu(L2)
print(L2)


# Max Pooling
# 슬라이딩 윈도우 크기 (2 x 2)  ksize = [1, 2, 2, 1]
# 두칸씩 움직여 stride = (2 x 2)
# zero padding 사용하므로 padding = SAME
# 출력 결과는 입력 받은 이미지의 반 => (?, 7, 7, 64)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
print(L2)

# Conv2 결과를 FC layer의 입력으로 주기위해 reshape = [-1, 3136]
# n 개 들어올거니 -1, 윈도우를 한줄로 펼치면 3136 (= 7 * 7 * 64)
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])  
print(L2)

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 1.3 Fully Connected (FC, Dense) Layer

- 크기변화
  - 입력이미지 784 -> L1 -> L2 -> FC 입력 3136 -> 출력 10 (0 ~ 9 사이 숫자이미지)

```
# Final FC layer
# 데이터 사이즈 변화
# 입력img 784 -> L1 -> L2 -> FC 입력 3136 -> 출력 10 (0 ~ 9 사이 숫자이미지)
# shape = [3136, 10] = [7 * 7 * 64, 10]
# xavier_initializer() 로 weight 초기화
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))  # 출력값 10 과 같아
logits = tf.matmul(L2, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 1.4 Training and Evaluation

```
print('Learning started.')
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
    # print('Accuracy:', sess.run(accuracy, feed_dict={
    #      X: mnist.test.images, Y: mnist.test.labels}))
      
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


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 1.5 MNIST CNN 전체코드

```
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
os.chdir('/home/testu/work') 
os.getcwd()

from tensorflow.examples.tutorials.mnist import input_data #/home/testu/work/MNIST_data 에 다운 받음
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(777)  # reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()


# hyper parameters
learning_rate   = 0.001
training_epochs = 2     # 15 는 싱글노드에선 넘 시간 많이 걸려
batch_size      = 100


# input place holders
# MNIST data image of shape 28 * 28 = 784
# None 은 몇개 instance 들어올지 명시 안하는 것. n개의 이미지 들어올 것이다
X = tf.placeholder(tf.float32, [None, 784])

# 입력 형식에 맞게 reshape
# n개의 데이터 (-1), h x w 크기 (28 x 28), 1개 색깔
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28 x 28 x 1 (black/white)

# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

# ------------------------------------------------------------
# Layer 1 
# 필터 사용해서 convolution 돌릴 것
# 필터 크기 (3 x 3), 1 색상, 32개 필터 사용
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))


# 필터는 한칸씩 움직이고 stride = (1 x 1)
# 입력 이미지와 같은 사이즈의 출력이미지를 만들어내 padding = SAME
# 단 32개 필터 사용하니 32개 이미지 생성됨 => (?, 28, 28, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print("Layer1\n{}".format(L1))

# ReLU 통과시킴
L1 = tf.nn.relu(L1)
print(L1)

# Max Pooling
# 슬라이딩 윈도우 크기 (2 x 2)  ksize = [1, 2, 2, 1]
# 두칸씩 움직여 stride = (2 x 2)
# zero padding 사용하므로 padding = SAME
# 출력 결과는 입력 받은 이미지의 반 => (?, 14, 14, 32)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
print(L1)

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# ------------------------------------------------------------
# Layer 2
# 입력 이미지 (conv layer1 로부터 받은) shape 는 (?, 14 x 14 x 32)
# 필터 사용해서 convolution 돌릴 것
# 필터 크기 (3 x 3 x 32), 64개 필터 사용
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))


# 필터는 한칸씩 움직이고 stride = (1 x 1)
# 입력 이미지와 같은 사이즈의 출력이미지를 만들어내 padding = SAME
# 단 64개 필터 사용하니 64개 이미지 생성됨 => (?, 14, 14, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print("Layer2\n{}".format(L2))

# ReLU 통과시킴 (출력 사이즈 변화는 없음)
L2 = tf.nn.relu(L2)
print(L2)


# Max Pooling
# 슬라이딩 윈도우 크기 (2 x 2)  ksize = [1, 2, 2, 1]
# 두칸씩 움직여 stride = (2 x 2)
# zero padding 사용하므로 padding = SAME
# 출력 결과는 입력 받은 이미지의 반 => (?, 7, 7, 64)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
print(L2)

# Conv2 결과를 FC layer의 입력으로 주기위해 reshape = [-1, 3136]
# n 개 들어올거니 -1, 윈도우를 한줄로 펼치면 3136 (= 7 * 7 * 64)
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])  
print(L2)

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

# ------------------------------------------------------------
# FC layer
# 데이터 사이즈 변화
# 입력img 784 -> L1 -> L2 -> FC 입력 3136 -> 출력 10 (0 ~ 9 사이 숫자이미지)
# shape = [3136, 10] = [7 * 7 * 64, 10]
# xavier_initializer() 로 weight 초기화
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))  # 출력값 10 과 같아
logits = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ------------------------------------------------------------
# Training and Evaluation
print('Learning started. It takes sometime.')
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        print('epoch {} processing'.format(epoch + 1))
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            print '.',
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
    # print('Accuracy:', sess.run(accuracy, feed_dict={
    #      X: mnist.test.images, Y: mnist.test.labels}))
      
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
epoch 1 processing...
('Epoch:', '0001', 'cost =', '0.395470340')
epoch 2 processing...
('Epoch:', '0002', 'cost =', '0.107678355')

Learning finished
('Accuracy: ', 0.97399998)
('Label: ', array([8]))
('Prediction: ', array([8]))
```


<br />
참조 링크

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-1-mnist_cnn.py#L58


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 2. Deep CNN

정확도 더 높아져
```
      input layer
           ↓
   Convolutional layer1
           ↓
     Pooling layer1
           ↓
   Convolutional layer2 
           ↓
     Pooling layer2
           ↓
   Convolutional layer3
           ↓
     Pooling layer3
           ↓
  Locally Connected layer
           ↓
   Fully Connected layer1
           ↓
   Fully Connected layer2
           ↓
      Output layer
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 2.2 Deep MNIST CNN 전체코드

메모리 문제로 실행 안되면 2.3 Deep MNIST CNN (Low Memory) 사용할것 

```
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
os.chdir('/home/testu/work') 
os.getcwd()

from tensorflow.examples.tutorials.mnist import input_data #/home/testu/work/MNIST_data 에 다운 받음
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(777)  # reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()


# hyper parameters
learning_rate   = 0.001
training_epochs = 2     # 15 는 싱글노드에선 넘 시간 많이 걸려
batch_size      = 100

# dropout rate: keep_prob =  0.5~0.75  (테스트 시 1.0)
keep_prob = tf.placeholder(tf.float32)


# input place holders
# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])

# 입력 형식에 맞게 reshape
# n개의 데이터 (-1), h x w 크기 (28 x 28), 1개 색깔
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28 x 28 x 1 (black/white)

# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])


# ------------------------------------------------------------
# Layer 1 
# input image (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# Conv     -> (?, 28, 28, 32)
# Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# ------------------------------------------------------------
# Layer 2
# input image (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 3
# input image (?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # flatten for FC layer
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])  
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 4: FC
# inputs 2048 (4 * 4 * 128) -> 625 outputs
W4 = tf.get_variable("W4", shape=[2048, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))  # 출력값 625 와 같아
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 5: Final FC
# inputs 625 -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))  # 출력값 10 과 같아
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ------------------------------------------------------------
# Training and Evaluation
print('Learning started. It takes sometime.')
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        print('epoch {} processing'.format(epoch + 1))
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            print '.',
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")


    # Test model and check accuracy
    # if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py
    # Test model
    h = tf.nn.softmax(logits)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.arg_max(h, 1), tf.arg_max(Y, 1)) # github 자료는 h 대신 logits (softmax 적용안된값) 썼어
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))
    # print('Accuracy:', sess.run(accuracy, feed_dict={
    #      X: mnist.test.images, Y: mnist.test.labels}))
      
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

<br />
참조 링크

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-2-mnist_deep_cnn.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### 2.3 Deep MNIST CNN (Low Memory) 전체코드

2.2 Deep MNIST CNN 실행 시 메모리 문제 생기는 경우 사용

```
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
os.chdir('/home/testu/work') 
os.getcwd()

from tensorflow.examples.tutorials.mnist import input_data #/home/testu/work/MNIST_data 에 다운 받음
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(777)  # reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()


# hyper parameters
learning_rate   = 0.001
training_epochs = 2    # 15 는 싱글노드에선 넘 시간 많이 걸려
batch_size      = 100

# dropout rate: keep_prob =  0.5~0.75  (테스트 시 1.0)
keep_prob = tf.placeholder(tf.float32)


# input place holders
# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])

# 입력 형식에 맞게 reshape
# n개의 데이터 (-1), h x w 크기 (28 x 28), 1개 색깔
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28 x 28 x 1 (black/white)

# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])


# ------------------------------------------------------------
# Layer 1 
# input image (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# Conv     -> (?, 28, 28, 32)
# Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# ------------------------------------------------------------
# Layer 2
# input image (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 3
# input image (?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # flatten for FC layer
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize  =[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])  
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 4: FC
# inputs 2048 (4 * 4 * 128) -> 625 outputs
W4 = tf.get_variable("W4", shape=[2048, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))  # 출력값 625 와 같아
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''


# ------------------------------------------------------------
# Layer 5: Final FC
# inputs 625 -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))  # 출력값 10 과 같아
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ------------------------------------------------------------
# Training and Evaluation
print('Learning started. It takes sometime.')
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(training_epochs):
        print('epoch {} processing'.format(epoch + 1))
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            print '.',
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            
            c, _ = sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")


    # Test model and check accuracy
    # if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py
    # Test model
    h = tf.nn.softmax(logits)
    
    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def evaluate(X_sample, y_sample, batch_size=512):
        """Run a minibatch accuracy op"""

        N = X_sample.shape[0]
        correct_sample = 0

        for i in range(0, N, batch_size):
            X_batch = X_sample[i: i + batch_size]
            y_batch = y_sample[i: i + batch_size]
            N_batch = X_batch.shape[0]

            feed = {
                X: X_batch,
                Y: y_batch,
                keep_prob: 1
            }

            correct_sample += sess.run(accuracy, feed_dict=feed) * N_batch

        return correct_sample / N

    print("\nAccuracy Evaluates")
    print("-------------------------------")
    print('Train Accuracy:', evaluate(mnist.train.images, mnist.train.labels))
    print('Test Accuracy:', evaluate(mnist.test.images, mnist.test.labels))


    # Get one and predict
    print("\nGet one and predict")
    print("-------------------------------")
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(h, 1), {X: mnist.test.images[r:r + 1], keep_prob: 1})
      

    plt.imshow(
         mnist.test.images[r:r + 1].reshape(28, 28),
         cmap='Greys',
         interpolation='nearest')
    plt.show()
```

실행 결과
```
Learning started. It takes sometime.
epoch 1 processing
. . . . . . . 
('Epoch:', '0001', 'cost =', '0.385817970')

epoch 2 processing
. . . . . . . 
('Epoch:', '0002', 'cost =', '0.097272345')

Learning finished

Accuracy Evaluates
-------------------------------
('Train Accuracy:', 0.98549090913425796)
('Test Accuracy:', 0.98519999933242797)

Get one and predict
-------------------------------
('Label: ', array([3]))
('Prediction: ', array([3])) 
```

<br />
참조 링크

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-X-mnist_cnn_low_memory.py


<br /><br />
