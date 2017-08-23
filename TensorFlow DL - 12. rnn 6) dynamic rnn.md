<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-6. Dynamic RNN


RNN의 강점은 sequence data 처리인데

지금까진 fixed sequence length 썼어

실제 예제에선 학습시킬 문자열 매번 다를 수 있다

이거 처리 방법 볼 것

<br />

참조 링크  

https://www.youtube.com/watch?v=aArdoSpdMEc&feature=youtu.be  
http://coolingoff.tistory.com/category/%5B%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%5D

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Different Sequence Length

이전까진 sequence는 정해진 크기로 사용  
그러나  번역같은 경우 정해진 sequence를 가지고 학습하기 어려움

(예)  
어떤 사람은 10자의 문장을 번역하고 싶은데 sequence가 15 라면 안맞음

```
         o        o        o        o        o 
         ↑        ↑        ↑        ↑        ↑
        [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] -> 
 입력    ↑        ↑        ↑        ↑        ↑

batch1:  h        e        l        l        o
         
batch2:  h        i
         
batch3:  w        h        y
         
  ...
 
```

기존의 방법은 CNN 처럼 padding 사용  

(예)  
sequence가 5 이고 입력할 문자가 hi이면 'h', 'i', <pad>, <pad>, <pad>

```
 h        e        l        l        o
 
 h        i      <pad>    <pad>    <pad>
 
 w        h        y      <pad>    <pad>
```
그런데 < pad > 에도 weight 적용되기때문에 어떤 값이 나와  
loss 계산 시 loss 함수에 영향미칠 수 있어서  
안좋은 결과 나올 수 있음  


==> Tensorflow 에선 각 배치의 길이를 알려줄 수 있어!

문자열을 줄 때 각 batch 마다 sequenc의 크기를 리스트에 저장가능

(예)
"hello" "hi" "why" 같은 경우 sequence_length = [5, 2, 3]


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Tensorflow: dynamic_rnn

dynamic_rnn 함수에서 sequence_length 인자로 조정할 수 있어

```
sequence_length=[5,3,4] 로 설정해보면,
outputs, _states = tf.nn.dynamic_rnn(
    cell, X_one_hot, initial_state=initial_state, sequence_length=[5,3,4], dtype=tf.float32)
```

결과
```

[[[-0.04720204 -0.0599511   0.10099947 -0.05461047 -0.01099665]
  [-0.10833281 -0.08143932  0.07744762 -0.04282264 -0.02629501]
  [-0.1025536   0.02003708  0.08273832 -0.02411628 -0.01970695]
  [-0.10984966  0.07807013  0.09081405 -0.01967948 -0.01748725]
  [-0.18398312 -0.07916182  0.07311723  0.00454873 -0.05470823]]  <-- 다 사용함

 [[-0.05238053 -0.04876425 -0.00061451 -0.02453336  0.00197438]
  [-0.13853857 -0.14233379 -0.01153345  0.01354709 -0.0423432 ]
  [-0.10130337 -0.00097406 -0.01684584  0.03078088 -0.01286638]
  [ 0.          0.          0.          0.          0.        ]   <-- 3 이후는 사용안함
  [ 0.          0.          0.          0.          0.        ]]

 [[-0.01673699  0.09656537  0.01489519  0.00161168  0.00773564]
  [-0.04099806  0.14449263  0.03371958 -0.00634202  0.00827252]
  [-0.11123932  0.03665368  0.03576175 -0.04223198 -0.00277512]
  [-0.16324286 -0.05752181  0.03917573 -0.05657914 -0.01601821]
  [ 0.          0.          0.          0.          0.        ]]] <-- 4 이후는 사용안함
  
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Simple Example

```
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.reset_default_graph()

sample   = "hello!"

# index -> char ['!', 'h', 'e', 'l', 'o']
char_set = list(set(sample))

# char -> index {'!': 0, 'e': 2, 'h': 1, 'l': 3, 'o': 4}
char_dic = {c: i for i, c in enumerate(char_set)}

batch1 = [char_dic[c] for c in "hello"] # [0, 1, 2, 2, 3]
batch2 = [char_dic[c] for c in "eolll"] # [1, 3, 2, 2, 2]
batch3 = [char_dic[c] for c in "lleel"] # [2, 2, 1, 1, 2]

batch_size      = 3
data_dim        = len(char_set)    # dictionary 크기, one hot size로 사용되니 입력 크기가 됨
hidden_size     = len(char_set)    # RNN (cell) output size
num_classes     = len(char_set)    # final output size (RNN or softmax, etc.)
sequence_length = 5
learning_rate   = 0.1

dataX = np.array([batch1, batch2, batch3], dtype=np.float32) 
dataY = np.array([char_dic[c] for c in "ello!"], dtype=np.float32)

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label
X_one_hot = tf.one_hot(dataX, num_classes)

# cell, num_unite = hidden_size (dictionary 사이즈와 같아) 
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X_one_hot, initial_state=initial_state, sequence_length=[5,3,4], dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1):
        print(outputs.eval())

```




