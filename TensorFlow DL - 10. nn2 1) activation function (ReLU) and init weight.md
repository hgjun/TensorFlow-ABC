<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 10-1. Better Non-linearity


10장의 큰 주제: Deep learning 잘하는 방법들

<br />
참조 링크  
https://www.youtube.com/watch?v=cKtg_fpw88c&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Vanishing Gradient 문제 (1986 - 2006)

NN 학계 2차 겨울 시작

layer 많으면 앞단 레이어의 값이 뒤에 거의 영향을 미치지 않아  
Backpropagation 계산 시 chain rule 하다보면  
0.01 x 0.03 x ...  
미분되어 계산된 소수점의 작은 값들이 레이어들 따라 계속 곱해지면 매우 작은값이 되  
경사도가 사라져 -> 학습하기 어렵다  

2 ~ 3 레이어 넘어가면 학습 안된다


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Geoffrey Hinton's Summary of Findings Up to Today

1) Our labeled datasets were thousands of times too small.
2) Our computers were millions of times too slow.
3) We initialized the weights in a stupid way.
4) We used the wrong type of non-linearity.

그중 4) 번 문제 때문, Sigmoid 를 잘 못 쓴 결과로 vanishing gradient 발생

SIgmoid 의 f(z) 값은 0 ~ 1 사이. 즉 항상 1보다 작은 값 쓰니  
작은 값 chain rule로 계속 곱해나가다 보면 최종 값은 매우 작은 값 되

==> ReLU 라는 activation function 쓰자!


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### ReLU (Rectified Linear Unit)

ReLU function: max(0, x)

z가 0보다 작으면 꺼, 크면 z값에 비례해서 linear 하게 키워

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1) 대신

layer1 = tf.relu(tf.matmul(X, W1) + b1) 사용!

(마지막 output layer 만 sigmoid 사용, 0 ~ 1의 출력을 만들어야 하니)

cost function 도 값 금방 떨어져, accuracy 좋음


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Activation Functions

Sigmoid: σ(x) = 1 / (1 + e^(-x))

tanh   : tanh(x)  -1 ~ 1 사이 값 나옴

ReLU   : max(0, x)


#### ReLU 응용

Leaky ReLU  max(0.1x, x)     x < 0 일때도 값 조금 주자

Maxout  max(wT1x + b1, wT2x + b)

ELU  
x < 0 일때 Leaky ReLU 는 0.1 로 고정시켰는데, 이걸 조금 더 조정할 수 있게  
f(x) = x  if x > 0  
f(x) = α(exp(x) -1)  if x ≤ 0


#### 사용빈도
Sigmoid 잘 안씀
 
tanh 가끔 사용

ReLU, Leaky ReLU 많이 사용


비교한 논문  
[Mishkin et al. 2015]  
Activation functions on CIFAR-10


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 10-2. Initialize Weights in a Smart Way


딥러닝 잘하는 방법 2) weight 초기값 어떻게 할 것인가 

vanishing gradient 문제

아까 Geoffrey Hinton's summary of findings up to today

1) Our labeled datasets were thousands of times too small.
2) Our computers were millions of times too slow.
3) We initialized the weights in a stupid way.
4) We used the wrong type of non-linearity.

4) 번 문제 ==> 좋은 Activation function 써서 해결 (ReLU)

두 번째 문제는 3) 때문  
같은 ReLU 를 써도 실험할 때마다 cost function 의  
cost (loss) 떨어지는 모양이 각각 달라져


(예)  
w 다 0 으로 주면  
x * w 인 layer 있을 때 x unit 부분은 다 0 되,  
backpropagation 으로 x 이전 unit 들도  다 0 되서  
기울기 값 사라져 


참조 링크  
https://www.youtube.com/watch?v=4rC0sWrp3Uw&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Initial Weight Values

weight 값 다 0 주면 안돼

- RBM (Restricted Boltzmann Machine) 으로 해결 (지금은 안쓰지만)
  - RBM으로 초기화시킨 네트웍이 Deep Belief Nets
  - Hinton et al. (2006) A Fast Learning Algorithm for Deep Belief Nets


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### RBM (Restricted Boltzmann Machine)

- RBM 구조
  - Symmetric Bipartite Graph 임
  - Restrition: no connecting within a layer (즉 bipartite graph 란 뜻)


Recreate input 입력을 재계산하기 위한 operation

<Forward 단> Encode  
layer1 입력 x1, x2, x3, ..  
w1, w2, w3, ... 곱해서 layer2 로 내보내


<Backward 단> Decode  
출력해서 받은 layer2 의 x 들에  
w1, w2, w3 ... 곱해서 거꾸로 layer 1로 보내 x1', x2', x3', ...


x1, x2, x3 과 x1', x2', x3' 을 비교  
이 둘의 차가 최저가 되도록 (두 값이 거의 같아지도록)  
weight 들을 조정!

KL divergence = compare actual to recreation

참조 링크  
https://deeplearning4j.org/kr/restrictedboltzmannmachine


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Deep Belief Network (2006)

RBM으로 초기화 값 잘 주면 된다

1) pre-traing step 으로 두 레이어를 RBM 학습
2) 모든 레이어로 RBM 학습 시켜
3) weight 설정돼

그 다음 x 넣고 학습 시켜  

(Find tuning, 학습이라고 안하고 이렇게 불러,  
이미 w 설정되어 있고 조금만 튜닝하면 되니)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Other Initialization Methods

굳이 RBM 안써도 된다. Simple methods 도 OK

weight 값이 'just right' 이면 돼 (not too small, not too big)  

노드의 입력 개수 (fan_in), 출력 개수 (fan_out) 에 비례해 weight 조절해주면 된다

- Xavier initialization
  - W = np.random.rndn(fan_in, fan_out) / np.sqrt(fan_in)  
  - RBM 과 비슷 혹은 조금 더 잘되더라
  - X. Glorot and Y.Bengio "Understanding the difficulty of training deep feedforward neural networks"
  - International conference on artificial intelligence and statistics, 2010

- He's initialization 
  - W = np.random.rndn(fan_in, fan_out) / np.sqrt(fan_in / 2)  
  - 모든 이미지넷의 오류 3%로 떨어짐
  - K. He (resnet 만든 사람, 홍콩 중문대 박사), X. Zhang, S. Ren, and J. Sun
  - "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," 2015


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Xavier Initialization in TensorFlow

```
W = tf.get_variable("W", shape=[784, 256],
           initializer=tf.contrib.layers.xavier_initializer())
```

```
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)
```

<br />
참조 링크  
https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Activation Functions and Initialization on CIFAR 10

비교한 논문 [Mishkin et al. 2015]

Xavier 등 써도 잘 되더라 실험 결과 보임


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Still an Active Area of Research

아직 perfect weight values 는 몰라

- Many new algorithms
  - Batch normalization, Layer sequential uniform variance, ...


여러개 직접해보고 그 중 잘 되는거 고르면 돼


<br /><br />














