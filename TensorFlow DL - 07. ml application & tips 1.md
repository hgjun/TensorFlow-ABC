<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 07. Application & Tips


I. 머신러닝 알고리즘 구현 시 팁들

1) Learning rate 조정하는 법
2) Data preprocessing 하는 법
3) Overfitting 방지하는 법!!


II. 머신러닝 기법 얼마나 잘 동작하는지 확인하는 방법  
Performance evaluation


참조 링크  
https://www.youtube.com/watch?v=1jPjVoDV_uo&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-1) Learning rate 조정하는 법

gradient descent 때 learning_rate 임의로 정해 실습했어
```
learning_rate = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypo), axis = 1))  # axid 대신 reduction_indices 써도 되더라
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)
```

learning_rate 잘 정하는게 중요
- Large learning rate
  - overshoot 너무 껑충껑충 뛰어, - + 왔다갔다, cost 줄어들지 않고 확커져서 밖으로 튀어나가, 값 줄여야
- Small learning rate
  - takes too long, stops at local minimum, 실행했더니 작게 변하더라하면 값 조금 올려야


Try several learning rates 특별한 답은 없어 (데이터, 실험 환경 따라 다 달라)  
Observe the cost function  
Check it goes down in a reasonable rate  
보통 0.01 로 시작, cost 함수 보면서 조절  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-2) Data preprocessing 하는 법

Data (X) preprocessing for gradient

x1 값 1~10 사이값, x2 값 1000 ~ 10000 사이 값이면  
w1, w2 도 단위차이 커  
learning rate가 w2 학습하는데 좋은 값이어도 w1에는 너무 커서 껑충껑충 뛰게 만들 수 있어  
normalize 필요!


original data -> zero-centered data -> normalized data

original data 가 skewed 되있으면 zero-centered data 해줘  
값이 너무 양수만 (혹은 음수만) 나오게 하는 문제 해결 가능

normalized data로 x1 x2 단위 같게 해주면 좋아

하는 방법  

- Standardization
  - x' = (x - μ) / σ 
  - μ [뮤] 평균
  -σ [시그마] 표준편차

```
x_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-3) Overfitting 방지하는 법

머신러닝의 가장 큰 문제!  
트레이닝 데이터는 잘 맞추나 실제 데이터 예측은 안맞아  


트레이닝 데이터를 많이 구하기 (많을수록 overfitting 줄일 수 )  
feature 개수 줄이기 (중복된거 줄이는 등)  
Regularization (일반화)  


#### Regularization  
weight 너무 큰 값 가지지 말자

2차원 상에 data 있을 때 이를 classification하는 decision boundary를   
너무 다 맞추려고 wavy하게 한다는 것은 weight도 크다는 의미  
weight 작게하면 decision boundary 선도 구불구불한게 점점 펴저  

cost function 에 하나 더 추가

L = 1/N Σ(D(S(WX + b), L) + λΣW^2

- λΣW^2
  - 각 w element들 제곱해서 다 합해줘
  - λ: 상수, regularization strength  (0이면 regularization 안하겠다, 클수록 이 값 영향커져)  

cost 함수 미분시켜 gradient descent 구할때 - ηΔL  
λΣW^2 도 미분 되면서 ΔL값 줄여줌

 
코드 예
```
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
cost = tf.reduce_mean(tf.square(h - Y)) + l2reg
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

(cf) 유석인 교수님 머신러닝 강좌  
chap 4. artificial neural network

마지막 부분 Alternative error functions
1) Penalize large weights (weight decay) 방법임


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Performance evaluation

training data로 -> ML model 학습시켜

테스트시 training data 그대로 쓰면 NG (당연히 다 맞다고 할 것)


data: 70% training set, 30% test set 으로 사용 [[1]]

[1]: http://www.holehouse.org/mlclass/10_Advice_for_applying_machine_learning.html

α 혹은 η[에타]: learning rate  
λ: regularization strength  

이런값 튜닝위해

training set 을 다시 training, validation 용으로 나눠  

training set으로 학습  
validation 으로 모의시험해서 α,η,λ 등 체크


(예)  
MINIST Dataset  

숫자 필기체 인식용 데이터 (우체국에서 우편번호 식별 위해)
(http://yann.lecun.com/exdb/mnist)

```
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```
도 트레이닝, 테스트 데이터로 나뉘어 있어  


(cf)  
Online learning 방법

데이터 많으면 한번에 다 사용하기 힘들어

- training set 100만개 -> Model 
  - 잘라서 학습
  - 10만개 -> Model 학습
  - 10만개 -> Model 추가 학습 (이전 10만 개 학습 결과로 학습된 상태)
  - 10만개 -> Model 추가 학습
  - ...


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Accuracy

실제 데이터 Y 값과 모델이 예측한값 비교

10개중 7개 맞추면 70%


