<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 06. Softmax Classification


참조 링크  
https://www.youtube.com/watch?v=VRnubDzIy3A&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### One-hot encoded vector

BInary logistic classification 일 때는 True/False 표현 위해  
Y 값으로 0, 1 사용하면 됐었어

Multinomial logistic classification 일 때는 One-hot encoded vector 필요  

(예) Y의 도메인 {A, B, C} 일때 이 세값을 one-hot encoded vector로 표현하면  
A = (1, 0, 0)  
B = (0, 1, 0)  
C = (0, 0, 1)  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax function

Linear model 을 사용해 hypothesis를 정의하면

H(X) = WX

H(X)의 값은 Label (실제 Y값) 과 바로 비교하기 어렵다  
따라서 One-hot encoded vector인 Y와 비교할 수 있도록  
Softmax function에 H(X)값을 입력해서 0 ~ 1 사이의 값으로 만들어준다  
확률값의 의미로 사용할 수 있게 된다!

즉 Softmax function의 역할  
H(X) Score 값을 Probability 값으로 만들어준다


Softmax function S(yi) = exp(yi) / Σexp(yj)

(예)
H(X) 를 y 로 사용해 표시해서,  
y1 = 2.0  
y2 = 1.0  
y3 = 0.1  

exp(2.0) = 7.389  
exp(1.0) = 2.718  
exp(0.1) = 1.105  

Σexp(yj) = (exp(2.0) + exp(1.0) + exp(0.1))  
          = 7.389 + 2.718 + 1.105 = 11.213

S(y1) = exp(2.0) / Σexp(yj) = 0.659  
S(y2) = exp(1.0) / Σexp(yj) = 0.242  
S(y3) = exp(0.1) / Σexp(yj) = 0.098  

S(y1) + S(y2) + S(y3) = 1.0  확률의 특성 (다 더하면 1) !!


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cross Entropy

계산 과정
```
[1]  X     input

    ↓     linear model WX + b

[2]  y     score (logit 이라고도 부름)

    ↓     softmax function S(y) = exp(yi) / Σexp(yj)

[3] S(y)   probability

    ↓     cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
           cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
           gradient descent W <- W + ΔW ,  ΔW = - ηΔL

[4] L      label (one-hot encoded vector)
```

softmax로 구한 학습결과값 S(y)와 실제값 L 을 비교하기위해 cross entropy를 사용한다

S(y)는 0 ~ 1 사이의 확률값으로 볼 수 있고  
L 도 one-hot encoded vector 이므로 0.0 or 1.0인 확률값으로 볼 수 있다  
S(y) 와 L 차이가 얼마나 나는지 보기 위해 cross entropy 사용

D(S, L) = D(S(WX + b), L) = - Σ Li log(Si)

* D( ) 의 D 는 distance
* D( ) 는 Symmetric 하지 않음 D(S, L) ≠D(L, S)

Semantic을 고려한 수식은   Σ Li * -log(Si) 임

(예1)  
Li 가 [1 0] 이고 Si 가 [0 0.99] 이면  
Li * -log(Si) = [1 0] ⊙ [∞ 0] = ∞  
큰 값이 되서 weight update 시키게 됨


(예2)  
Li 가 [1 0] 이고 Si 가 [0.99 0] 이면  
Li * -log(Si) = [1 0] ⊙ [0 ∞] = 0


참조 링크  
https://www.youtube.com/watch?v=tRsSi_sqXjI
https://www.youtube.com/watch?v=x449QQDhMDE


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cross Entropy vs. Mean Squared Error

학습 시킬 때 MSE 대신 Cross Entropy 사용하는 이유


이전 binary logisitc classifiction은 False 가 아니면 True였어  
그러니 틀린것만 봐서 Mean squared error 를 사용해  
틀린 정도에 따라 weight를 update 시켰어


그런데 multinomial logistic classification은 틀린것만 봐선 안돼  

Li = (1, 0, 0)  

S(yi) = (0, 0.5, 0) 이면 MSE로도 틀린 정도에 따라 업데이트 가능  
S(yi) = (0.6, 0, 0) 이면 error 없는 걸로 봐서 업데이트 하지 않아

하지만 (0.7, 0, 0) 이라면 Li와 더 비슷한 값이기에 맞는 것에 대해서도 업데이트 해줘야해!

그래서 D(S, L) 로 두 값의 distance를 측정해서  
얼마나 많이 틀렸는지, 얼마나 정확하게 맞았는지를 보고
더 잘 업데이트 해주기 위한 것!


(cf)
update 를 더 잘해주는 측면에서 비슷한 예로  
sigmoid function이 x축이 매우 크거나 작으면 변화의 정도가 작아 update 잘 되지 않아  
그런데 LeRU 는 변화 일정해 x축 매우 크거나 작아도 update 상대적으로 더 잘 해주는 이점 있다

<br />
참조 링크  
https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/  
http://funmv2013.blogspot.kr/2017/01/cross-entropy.html


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost function: Cross Entropy 

Loss L:

L = 1/N * Σ D(S(WXi + b), Li)

W <- W + ΔW

ΔW = - ηΔL

(Loss function의 미분은 tensor flow에서는 알아서 해줌)  
Σ는 N개의 training set의 D(Si, Li) 를 다 합쳐준다.
 

#### Gradient descent

이거로 Min cost 구하기 위해 미분해서 ΔW = - ηΔL 로  
값을 계속 업데이트 해 나간다

(cf) Logistic cost vs. Cross Entropy

Logistic cost 

C(H(x), y) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

Cost(W) = - 1/m Σ( y log( H(x) ) + (1 - y) log( 1 - H(x) ) )

Logistic cost 와 Cross entropy는 같은 의미로 볼 수 있다


```
logistic 에서는 y값이 0 또는 1 이있는데,
이를 one-hot encoding 벡터로 바꿔서 cross entropy를 적용해 보세요.
0=>[1, 0], 1=>[0, 1].

logistic 을 풀자면 
-log(H(x))   : y = 1 => [0, 1]
-log(1-H(x)) : y = 0 => [1, 0]

cross entropy 를 풀자면
sigma(Li * -log(Si))

y = L, H(x) = S 이므로

L:[0, 1], S:H(x)
sigma([0, 1] ( * ) -log[0, 1]) = 0

L:[1, 0], S:1-H(x)
sigma([1, 0] ( * ) -log[1-0, 1-1]) = sigma([1,0] ( * ) -log[1,0]) = 0

이 대입으로 보면 logistic cost & cross entropy 는 같은 의미입니다.
------------------------------------------------------------------
 https://www.youtube.com/watch?v=jMU9G5WEtBc&feature=youtu.be
``` 
 
 <br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax Classifier

```
[1]  X     input

    ↓     linear model     tf.matmul(X, W) + b

[2]  y     score (logit)

    ↓     softmax function S(y) = exp(yi) / Σexp(yj)
                            hypo = tf.nn.softmax(tf.matmul(X, W) + b)
                            어떤 label이 될건가에 대한 확률값으로 만들어줌

[3] S(y)   probability

    ↓     cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
           cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
           gradient descent W <- W + ΔW ,  ΔW = - ηΔL
           
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

nb_classes = 3   # nb_는 number of 라는 의미로 쓴 prefix 인듯

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape = [None, 4])
Y = tf.placeholder(tf.float32, shape = [None, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight') # X feature 개수(4), Y 개수(3) -> shape[4, 3] 4개 들어와서 3개 나와
b = tf.Variable(tf.random_normal([nb_classes]), name='bias') # 나가는 값 (Y)의 개수

# hypothesis H(X) = exp(yi) / Σexp(yj)
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)
h = tf.nn.softmax(tf.matmul(X, W) + b)

# cost function: cross entropy cost/loss
# cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
# cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis = 1))

# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
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
tf에서 사용되는 함수의 argument 이름  
axis, dimension, dim, indices --> 다 axis 로 통일됨

https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-1-softmax_classifier.py


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 06-2. Fancy Softmax Classifier

cross_entropy, one_hot, 특히 reshape 이용해 구현하기


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax Classifier

```
[1]  X     input

    ↓     linear model     tf.matmul(X, W) + b

[2]  y     score (logit)

    ↓     softmax function S(y) = exp(yi) / Σexp(yj)
                            hypo = tf.nn.softmax(tf.matmul(X, W) + b)
                            어떤 label이 될건가에 대한 확률값으로 만들어줌

[3] S(y)   probability

    ↓     cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
           cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
           gradient descent W <- W + ΔW ,  ΔW = - ηΔL
           
           cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypo), axis = 1))
           optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
           
[4] L      label (one-hot encoded vector)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### softmax_cross_entropy_with_logits

아까 이렇게 썼었어
```
# softmax
logits = tf.matmul(X, W) + b    // logit 혹은 score
hypo   = tf.nn.softmax(logits)  // score 에서 probability 값으로 변경

# cost function
cost = tf.reduce_mean( -tf.reduce_sum(Y * tf.log(hypo), axis = 1))  # Y 는 one-hot encoded
```


- tf.nn.softmax_cross_entropy_with_logits
  - 위의 구문을 더 간소하게 만들어주는 tf 함수
  - softmax, cross entropy 역할 해줌 (이름 보면 알 수 있듯)
  - 두 입력값 필요: logit (softmax 들어가기 전의 값), Y 값

```
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost   = tf.reduce_mean(cost_i)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Numpy: zip

zip 함수는 동일한 갯수의 요소값을 갖는 시퀀스 자료형을 묶어주는 역할

```
import numpy as np
print(zip([1,2,3], [4,5,6]))
# [(1, 4), (2, 5), (3, 6)]

print(zip([1,2,3], [4,5,6], [7,8,9]))
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

print(zip("abc", "def"))
# [('a', 'd'), ('b', 'e'), ('c', 'f')]


# for zip 예제1
a = [1,2,3,4,5]
b = ['a','b','c','d','e']
 
for x,y in zip (a,b):
  print(x,y)
# 1 a
# 2 b
# 3 c
# 4 d 
# 5 e

  
# for zip 예제2
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

softmax_cross_entropy_with_logits 함수 연습으로  
실제 예제 사용해보기 (동물 데이터)

data03_zoo.csv 데이터 파일은  
C:\Users\(사용자 계정이름)\Docker\work 에 저장해서 사용

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

# Y 아직 one-hot encoded 되기 전 상태임
# Y는 0 ~ 6의 값: birds, insect, fishes, amphibians, reptiles, mammals
print(x_data.shape, y_data.shape)

nb_classes = 7  # Y는 7개의 class로 구성될 것이다 (0 ~ 6)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 16]) # X 속성 개수 16개
Y = tf.placeholder(tf.int32  , shape=[None, 1])  # Y 개수 1개, 주의!! int32 써야!!!
Y_one_hot = tf.one_hot(Y, nb_classes)            # Y 개수 7개 one-hot encoded 된 후
Y_one_hot_tmp = Y_one_hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
Y_one_hot_reshaped = Y_one_hot
print("Y", Y)                        # shape = (?, 1)
print("one_hot", Y_one_hot_tmp)      # shape = (?, 1, 7) 
print("reshape", Y_one_hot_reshaped) # shape = (?, 7)

# 그런데 tf.one_hot 함수는 n차원 자료 입력하면 n + 1차원 결과를 출력해 -> reshape 필요
# (예) 데이터 레코드 개수 2 이고, 첫번째 줄의 Y = 0,  두번째 줄 Y = 3 이라면
#Y = np.array( [[0], [3]])                           # rank 2 (2차원)   shape = (2, 1)
#print(Y, Y.shape)
#Y_one_hot = tf.one_hot(Y, 6)                        # rank 3 (3차원)   shape = (2, 1, 6)
#print(Y_one_hot, Y_one_hot.shape)                   # [ [[1 0 0 0 0 0]], [[0 0 0 1 0 0 0]] ]
#Y_one_hot_reshaped = tf.reshape(Y_one_hot, [-1, 6]) # shape = (2, 6)
#print(Y_one_hot_reshaped, Y_one_hot_reshaped.shape) # [ [1 0 0 0 0 0], [0 0 0 1 0 0 0] ]

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight') # X feature 개수(4), Y 개수(3) -> shape[4, 3] 4개 들어와서 3개 나와
b = tf.Variable(tf.random_normal([nb_classes]), name='bias') # 나가는 값 (Y)의 개수

# tf.nn.softmax computes softmax activations
# hypothesis H(X) = exp(yi) / Σexp(yj)
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)
logits = tf.matmul(X, W) + b
h = tf.nn.softmax(logits)

# cost function: cross entropy cost/loss
# cost function    L = 1/N * Σ D(S(WXi + b), Li)    cost function L은 loss 의미 
# cross entropy    D(S, L) = - Σ Li log(Si)         D(S, Li)의 L은 label 의미
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis = 1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost   = tf.reduce_mean(cost_i)

# Minimize: small learning rate
# W <- W - η* ∂Cost(W) / ∂W
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# Accuracy computation
# (참고) 이전 binary logistic classifier 에서의 prediction 코드
# prediction         = tf.cast(h > 0.5, dtype = tf.float32) # 1.0 or 0.0
# correct_prediction = tf.equal(prediction, Y)
# accuracy  = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
# 10 개 중 5개 true로 맞췄으면 accuracy = true개수 / 전체 개수 = 0.5 될 것.

prediction         = tf.argmax(h, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # 실제 Y와 예측이 맞은 개수
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
	        # Step:  1800	Cost: 0.322	Accuracy: 93.07%  형태로 출력

	pred = sess.run(prediction, feed_dict = {X: x_data, Y: y_data})
	# y_data: (N,1) = flatten => (N, ) matches pred.shape
	for p, y in zip(pred, y_data.flatten()):  # flatten 쓰면 [ [1], [2] ] -> [1, 2] 처럼 돼 (R의 unlist 비슷)
		print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
		# [True] Prediction: 4 True Y: 4
		# [False] Prediction: 0 True Y: 2  형태로 출력
		
	# (참고) 이전 binary logistic classifier 에서의 Accuracy report 코드
	#hypo, 예측한값, accuracy(예측한 결과가 실제 Y랑 몇 개 같나) 
	#hy, c, a = sess.run([h, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
	#print("\nHypothesis: ", hy,"\Correct: ", c,"\Accuracy: ", a)
```

<br />
참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-2-softmax_zoo_classifier.py


