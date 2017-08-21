<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 06. Softmax Classification


���� ��ũ  
https://www.youtube.com/watch?v=VRnubDzIy3A&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### One-hot encoded vector

BInary logistic classification �� ���� True/False ǥ�� ����  
Y ������ 0, 1 ����ϸ� �ƾ���

Multinomial logistic classification �� ���� One-hot encoded vector �ʿ�  

(��) Y�� ������ {A, B, C} �϶� �� ������ one-hot encoded vector�� ǥ���ϸ�  
A = (1, 0, 0)  
B = (0, 1, 0)  
C = (0, 0, 1)  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax function

Linear model �� ����� hypothesis�� �����ϸ�

H(X) = WX

H(X)�� ���� Label (���� Y��) �� �ٷ� ���ϱ� ��ƴ�  
���� One-hot encoded vector�� Y�� ���� �� �ֵ���  
Softmax function�� H(X)���� �Է��ؼ� 0 ~ 1 ������ ������ ������ش�  
Ȯ������ �ǹ̷� ����� �� �ְ� �ȴ�!

�� Softmax function�� ����  
H(X) Score ���� Probability ������ ������ش�


Softmax function S(yi) = exp(yi) / ��exp(yj)

(��)
H(X) �� y �� ����� ǥ���ؼ�,  
y1 = 2.0  
y2 = 1.0  
y3 = 0.1  

exp(2.0) = 7.389  
exp(1.0) = 2.718  
exp(0.1) = 1.105  

��exp(yj) = (exp(2.0) + exp(1.0) + exp(0.1))  
          = 7.389 + 2.718 + 1.105 = 11.213

S(y1) = exp(2.0) / ��exp(yj) = 0.659  
S(y2) = exp(1.0) / ��exp(yj) = 0.242  
S(y3) = exp(0.1) / ��exp(yj) = 0.098  

S(y1) + S(y2) + S(y3) = 1.0  Ȯ���� Ư�� (�� ���ϸ� 1) !!


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cross Entropy

��� ����
```
[1]  X     input

    ��     linear model WX + b

[2]  y     score (logit �̶�� �θ�)

    ��     softmax function S(y) = exp(yi) / ��exp(yj)

[3] S(y)   probability

    ��     cost function    L = 1/N * �� D(S(WXi + b), Li)    cost function L�� loss �ǹ� 
           cross entropy    D(S, L) = - �� Li log(Si)         D(S, Li)�� L�� label �ǹ�
           gradient descent W <- W + ��W ,  ��W = - ���L

[4] L      label (one-hot encoded vector)
```

softmax�� ���� �н������ S(y)�� ������ L �� ���ϱ����� cross entropy�� ����Ѵ�

S(y)�� 0 ~ 1 ������ Ȯ�������� �� �� �ְ�  
L �� one-hot encoded vector �̹Ƿ� 0.0 or 1.0�� Ȯ�������� �� �� �ִ�  
S(y) �� L ���̰� �󸶳� ������ ���� ���� cross entropy ���

D(S, L) = D(S(WX + b), L) = - �� Li log(Si)

* D( ) �� D �� distance
* D( ) �� Symmetric ���� ���� D(S, L) ��D(L, S)

Semantic�� ����� ������   �� Li * -log(Si) ��

(��1)  
Li �� [1 0] �̰� Si �� [0 0.99] �̸�  
Li * -log(Si) = [1 0] �� [�� 0] = ��  
ū ���� �Ǽ� weight update ��Ű�� ��


(��2)  
Li �� [1 0] �̰� Si �� [0.99 0] �̸�  
Li * -log(Si) = [1 0] �� [0 ��] = 0


���� ��ũ  
https://www.youtube.com/watch?v=tRsSi_sqXjI
https://www.youtube.com/watch?v=x449QQDhMDE


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cross Entropy vs. Mean Squared Error

�н� ��ų �� MSE ��� Cross Entropy ����ϴ� ����


���� binary logisitc classifiction�� False �� �ƴϸ� True����  
�׷��� Ʋ���͸� ���� Mean squared error �� �����  
Ʋ�� ������ ���� weight�� update ���׾�


�׷��� multinomial logistic classification�� Ʋ���͸� ���� �ȵ�  

Li = (1, 0, 0)  

S(yi) = (0, 0.5, 0) �̸� MSE�ε� Ʋ�� ������ ���� ������Ʈ ����  
S(yi) = (0.6, 0, 0) �̸� error ���� �ɷ� ���� ������Ʈ ���� �ʾ�

������ (0.7, 0, 0) �̶�� Li�� �� ����� ���̱⿡ �´� �Ϳ� ���ؼ��� ������Ʈ �������!

�׷��� D(S, L) �� �� ���� distance�� �����ؼ�  
�󸶳� ���� Ʋ�ȴ���, �󸶳� ��Ȯ�ϰ� �¾Ҵ����� ����
�� �� ������Ʈ ���ֱ� ���� ��!


(cf)
update �� �� �����ִ� ���鿡�� ����� ����  
sigmoid function�� x���� �ſ� ũ�ų� ������ ��ȭ�� ������ �۾� update �� ���� �ʾ�  
�׷��� LeRU �� ��ȭ ������ x�� �ſ� ũ�ų� �۾Ƶ� update ��������� �� �� ���ִ� ���� �ִ�

<br />
���� ��ũ  
https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/  
http://funmv2013.blogspot.kr/2017/01/cross-entropy.html


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost function: Cross Entropy 

Loss L:

L = 1/N * �� D(S(WXi + b), Li)

W <- W + ��W

��W = - ���L

(Loss function�� �̺��� tensor flow������ �˾Ƽ� ����)  
�Ҵ� N���� training set�� D(Si, Li) �� �� �����ش�.
 

#### Gradient descent

�̰ŷ� Min cost ���ϱ� ���� �̺��ؼ� ��W = - ���L ��  
���� ��� ������Ʈ �� ������

(cf) Logistic cost vs. Cross Entropy

Logistic cost 

C(H(x), y) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

Cost(W) = - 1/m ��( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

Logistic cost �� Cross entropy�� ���� �ǹ̷� �� �� �ִ�

<br />
```
logistic ������ y���� 0 �Ǵ� 1 ���ִµ�,
�̸� one-hot encoding ���ͷ� �ٲ㼭 cross entropy�� ������ ������.
0=>[1, 0], 1=>[0, 1].

logistic �� Ǯ�ڸ� 
-log(H(x))   : y = 1 => [0, 1]
-log(1-H(x)) : y = 0 => [1, 0]

cross entropy �� Ǯ�ڸ�
sigma(Li * -log(Si))

y = L, H(x) = S �̹Ƿ�

L:[0, 1], S:H(x)
sigma([0, 1] ( * ) -log[0, 1]) = 0

L:[1, 0], S:1-H(x)
sigma([1, 0] ( * ) -log[1-0, 1-1]) = sigma([1,0] ( * ) -log[1,0]) = 0

�� �������� ���� logistic cost & cross entropy �� ���� �ǹ��Դϴ�.
------------------------------------------------------------------
 https://www.youtube.com/watch?v=jMU9G5WEtBc&feature=youtu.be
``` 
 
 <br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax Classifier

```
[1]  X     input

    ��     linear model     tf.matmul(X, W) + b

[2]  y     score (logit)

    ��     softmax function S(y) = exp(yi) / ��exp(yj)
                            hypo = tf.nn.softmax(tf.matmul(X, W) + b)
                            � label�� �ɰǰ��� ���� Ȯ�������� �������

[3] S(y)   probability

    ��     cost function    L = 1/N * �� D(S(WXi + b), Li)    cost function L�� loss �ǹ� 
           cross entropy    D(S, L) = - �� Li log(Si)         D(S, Li)�� L�� label �ǹ�
           gradient descent W <- W + ��W ,  ��W = - ���L
           
           cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypo), axis = 1)) 
           optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
           
[4] L      label (one-hot encoded vector)
```



```
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
# Training data

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]] # one-hot endoded

nb_classes = 3   # nb_�� number of ��� �ǹ̷� �� prefix �ε�

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape = [None, 4])
Y = tf.placeholder(tf.float32, shape = [None, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight') # X feature ����(4), Y ����(3) -> shape[4, 3] 4�� ���ͼ� 3�� ����
b = tf.Variable(tf.random_normal([nb_classes]), name='bias') # ������ �� (Y)�� ����

# hypothesis H(X) = exp(yi) / ��exp(yj)
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)
h = tf.nn.softmax(tf.matmul(X, W) + b)

# cost function: cross entropy cost/loss
# cost function    L = 1/N * �� D(S(WXi + b), Li)    cost function L�� loss �ǹ� 
# cross entropy    D(S, L) = - �� Li log(Si)         D(S, Li)�� L�� label �ǹ�
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis = 1))

# Minimize: small learning rate
# W <- W - ��* ��Cost(W) / ��W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(2001):
		sess.run([train], feed_dict = {X: x_data, Y: y_data})
		if step % 200 == 0:
			print(step, "Cost: ", sess.run([cost], feed_dict = {X: x_data, Y: y_data}))
	
	#for step in range(10001):
	#	cost_val, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
	#	if step % 200 == 0:
	#		print(step, "Cost: ", cost_val)
			
	# Testing & one-hot encoding
	a = sess.run(h, feed_dict = {X: [[1, 11, 7, 9]]})
	print(a, sess.run(tf.arg_max(a, 1)))
	
	print('--------------')

	b = sess.run(h, feed_dict = {X: [[1, 3, 4, 3]]})
	print(b, sess.run(tf.arg_max(b, 1)))

	print('--------------')

	c = sess.run(h, feed_dict = {X: [[1, 1, 0, 1]]})
	print(c, sess.run(tf.arg_max(c, 1)))

	print('--------------')

	all = sess.run(h, feed_dict = {X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
	print(all, sess.run(tf.arg_max(all, 1)))
```


(cf)
tf���� ���Ǵ� �Լ��� argument �̸�  
axis, dimension, dim, indices --> �� axis �� ���ϵ�

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-1-softmax_classifier.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 06-2. Fancy Softmax Classifier

cross_entropy, one_hot, Ư�� reshape �̿��� �����ϱ�


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax Classifier

```
[1]  X     input

    ��     linear model     tf.matmul(X, W) + b

[2]  y     score (logit)

    ��     softmax function S(y) = exp(yi) / ��exp(yj)
                            hypo = tf.nn.softmax(tf.matmul(X, W) + b)
                            � label�� �ɰǰ��� ���� Ȯ�������� �������

[3] S(y)   probability

    ��     cost function    L = 1/N * �� D(S(WXi + b), Li)    cost function L�� loss �ǹ� 
           cross entropy    D(S, L) = - �� Li log(Si)         D(S, Li)�� L�� label �ǹ�
           gradient descent W <- W + ��W ,  ��W = - ���L
           
           cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypo), axis = 1))
           optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
           
[4] L      label (one-hot encoded vector)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### softmax_cross_entropy_with_logits

�Ʊ� �̷��� �����
```
# softmax
logits = tf.matmul(X, W) + b    // logit Ȥ�� score
hypo   = tf.nn.softmax(logits)  // score ���� probability ������ ����

# cost function
cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypo), axis = 1))  # Y �� one-hot encoded
```


- tf.nn.softmax_cross_entropy_with_logits
  - ���� ������ �� �����ϰ� ������ִ� tf �Լ�
  - softmax, cross entropy ���� ���� (�̸� ���� �� �� �ֵ�)
  - �� �Է°� �ʿ�: logit (softmax ���� ���� ��), Y ��

```
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost   = tf.reduce_mean(cost_i)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Numpy: zip

zip �Լ��� ������ ������ ��Ұ��� ���� ������ �ڷ����� �����ִ� ����

```
import numpy as np
print(zip([1,2,3], [4,5,6]))
# [(1, 4), (2, 5), (3, 6)]

print(zip([1,2,3], [4,5,6], [7,8,9]))
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

print(zip("abc", "def"))
# [('a', 'd'), ('b', 'e'), ('c', 'f')]


# for zip ����1
a = [1,2,3,4,5]
b = ['a','b','c','d','e']
 
for x,y in zip (a,b):
  print(x,y)
# 1 a
# 2 b
# 3 c
# 4 d 
# 5 e

  
# for zip ����2
t = np.array( [[0], [3]])
print(t)            #  [[0], [3]]
print(t.flatten())  # [0 3]

v = [1,2]
for p, r in zip(v, t.flatten()):
print(p, r)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Animal classification

softmax_cross_entropy_with_logits �Լ� ��������  
���� ���� ����غ��� (���� ������)

data03_zoo.csv ������ ������  
C:\Users\(����� �����̸�)\Docker\work �� �����ؼ� ���

```
import tensorflow as tf
import numpy as np
import os
os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 1) Build a graph using tf operations
xy = np.loadtxt('data03_zoo.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Y ���� one-hot encoded �Ǳ� �� ������
# Y�� 0 ~ 6�� ��: birds, insect, fishes, amphibians, reptiles, mammals
print(x_data.shape, y_data.shape)

nb_classes = 7  # Y�� 7���� class�� ������ ���̴� (0 ~ 6)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 16]) # X �Ӽ� ���� 16��
Y = tf.placeholder(tf.int32  , shape=[None, 1])  # Y ���� 1��, ����!! int32 ���!!!
Y_one_hot = tf.one_hot(Y, nb_classes)            # Y ���� 7�� one-hot encoded �� ��
Y_one_hot_tmp = Y_one_hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
Y_one_hot_reshaped = Y_one_hot
print("Y", Y)                        # shape = (?, 1)
print("one_hot", Y_one_hot_tmp)      # shape = (?, 1, 7) 
print("reshape", Y_one_hot_reshaped) # shape = (?, 7)

# �׷��� tf.one_hot �Լ��� n���� �ڷ� �Է��ϸ� n + 1���� ����� ����� -> reshape �ʿ�
# (��) ������ ���ڵ� ���� 2 �̰�, ù��° ���� Y = 0,  �ι�° �� Y = 3 �̶��
#Y = np.array( [[0], [3]])                           # rank 2 (2����)   shape = (2, 1)
#print(Y, Y.shape)
#Y_one_hot = tf.one_hot(Y, 6)                        # rank 3 (3����)   shape = (2, 1, 6)
#print(Y_one_hot, Y_one_hot.shape)                   # [ [[1 0 0 0 0 0]], [[0 0 0 1 0 0 0]] ]
#Y_one_hot_reshaped = tf.reshape(Y_one_hot, [-1, 6]) # shape = (2, 6)
#print(Y_one_hot_reshaped, Y_one_hot_reshaped.shape) # [ [1 0 0 0 0 0], [0 0 0 1 0 0 0] ]

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight') # X feature ����(4), Y ����(3) -> shape[4, 3] 4�� ���ͼ� 3�� ����
b = tf.Variable(tf.random_normal([nb_classes]), name='bias') # ������ �� (Y)�� ����

# tf.nn.softmax computes softmax activations
# hypothesis H(X) = exp(yi) / ��exp(yj)
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)
logits = tf.matmul(X, W) + b
h = tf.nn.softmax(logits)

# cost function: cross entropy cost/loss
# cost function    L = 1/N * �� D(S(WXi + b), Li)    cost function L�� loss �ǹ� 
# cross entropy    D(S, L) = - �� Li log(Si)         D(S, Li)�� L�� label �ǹ�
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis = 1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
# W <- W - ��* ��Cost(W) / ��W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# Accuracy computation
# (����) ���� binary logistic classifier ������ prediction �ڵ�
# prediction         = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
# correct_prediction = tf.equal(prediction, Y)
# accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
# 10 �� �� 5�� true�� �������� accuracy = true���� / ��ü ���� = 0.5 �� ��.

prediction         = tf.argmax(h, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # ���� Y�� ������ ���� ����
accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))

# 2) Feed data and run graph (operation) using sess.run(op)
# 3) Update variables in the graph (and return values)
with tf.Session() as sess:                       # launch graph in a session
	sess.run(tf.global_variables_initializer())  # init global variables
	
	for step in range(2000):
		sess.run([train], feed_dict = {X: x_data, Y: y_data})
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  ���·� ���

	pred = sess.run(prediction, feed_dict = {X: x_data, Y: y_data})
	# y_data: (N,1) = flatten => (N, ) matches pred.shape
	for p, y in zip(pred, y_data.flatten()):  # flatten ���� [ [1], [2] ] -> [1, 2] ó�� �� (R�� unlist ���)
		print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
		# [True] Prediction: 4 True Y: 4
		# [False] Prediction: 0 True Y: 2  ���·� ���
		
	# (����) ���� binary logistic classifier ������ Accuracy report �ڵ�
	#hypo, �����Ѱ�, accuracy(������ ����� ���� Y�� �� �� ����) 
	#hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	#print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```

<br />
���� ��ũ  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-2-softmax_zoo_classifier.py


