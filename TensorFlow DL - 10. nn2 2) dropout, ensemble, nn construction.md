<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 10-3. NN Dropout and Model Ensemble


10장의 큰 주제: Deep learning 잘하는 방법들

<br />
참조 링크  
https://www.youtube.com/watch?v=wTxMsp22llc&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Overfitting

너무 tr data 에 맞추려다보면 오히려 test data 로는 예측 accuracy 떨어져

Y축 error, X축 weight개수, layer 개수 그래프에서  
tr data 용 error 계속 떨어져  
test data는 떨어지다 어느 순간 다시 올라가 (overfitting)


=> 해결 방법
1) tr data 많이 둔다
2) feature 개수 줄인다 (deep learning 에선 굳이 이럴 필욘 없지만)
3) Regularization
4) Neural Nets 에선 하나 더 있어: Dropout


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Regularization

weight 넘 큰 값 갖게 하지 말자  
(TensorFlow ex - lab07. application & tips (a 이론) 파일 참조)

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


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Dropout

A Simple Way to Prevent Neural Networks from Overfitting [Srivastava et al. 2014]

randomly set som neurons to zero in the forward pass  
학습 시 몇 개 node 연결 끊어서 학습시키지 않게 해


(예)
고양이 판별  
unit 1 (귀 있나?)            X  
unit 2 (꼬리 있나?)  
unit 3 (털 있나?)            X  
unit 4 (발톱 있나?)  
unit 5 (말썽부리게 생겼나?)  X

몇 개 죽이고 일부만 가지고 학습시켜 (unit 2, unit 4 -> cat score)

그다음 다 이용해서 다시 학습시켜

dropout rate 는 보통 0.5 ~ 0.7 사용
```
dropout_rate = tf.placeholder("float")
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1)
# L1 하나 더 만들어, dropout 레이어에 보내고 이걸 다음 레이어로 보내
L1  = tf.nn.dropout(L1, dropout_rate)


Train:
sess.run(optimizer, feed_dict = {X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
```

#### 주의!!
실제 사용(혹은 평가)할 때는 unit 다 사용해야!! dropout_rate = 1

```
# Evaluation:
print "Accuracy: ", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1})
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Ensemble

dt set -> tr set 만들어

k 개 learning model 만들어 (9 layers) 각각 tr set 입력으로 넣고 학습 (tr set 동일한 하나로 써도 무방)
model 마다 결과 조금씩 다르게 나올 것  
마지막에 model k개 합쳐  
Ensemble Prediction 만들어  

(마치 여러명의 전문에게 물어보는 식)


이런식으로 학습시키면 성능 2 ~ 4 % 향상


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 10-4. NN Construction (like LEGO Playing)


<br />
참조 링크  
https://www.youtube.com/watch?v=YHsbHjTBx9Q&feature=youtu.be

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Feedforward Neural Network

layer 1개, 2개, ... 원하는 대로 쌓아서 사용하면되  
-> 굉장히 다양한 구조 가능


#### Fast Forward [Resnet, He et al.]

두단 앞으로 보내

```
x - [ ] - [ ] - [ ] - [ ] - [ ] - [ ] -  y-hat  
        └----------┘    └----------┘
```


#### Split & Merge

나눠서 처리하고 다시 합치거나,
```
            [ ] - [ ] - [ ]  
x -> [ ]  <                 > [ ] - [  ] -  y-hat  
            [ ] - [ ] - [ ]  
```         

각각 처리하고 후에 합치거나, (Convolutional NN)
```
x1 [ ] - [ ] - [ ]  ┐
x2 [ ] - [ ] - [ ]  ┼-- [ ] - [ ] -  y-hat
x3 [ ] - [ ] - [ ]  ┘          
```

#### Recurrent Network (RNN)

앞 뿐만 아니라, 옆으로도 나가자
```
y-hat
 ↑       ↑       ↑
 [ ]  →  [ ]  →  [ ]
 ↑       ↑       ↑
 [ ]  →  [ ]  →  [ ]
 ↑       ↑       ↑
 [ ]  →  [ ]  →  [ ]
 ↑       ↑       ↑
 x1       x2       x3   
```



