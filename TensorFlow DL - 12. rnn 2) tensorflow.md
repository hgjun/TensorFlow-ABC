<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-2. RNN TensorFlow

<br />

참조 링크  
https://www.youtube.com/watch?v=B5GtZuUvujQ&feature=youtu.be
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-0-rnn_basics.ipynb

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### RNN in TensorFlow

```
           y
           ↑
           │  ┌───┐  지금 cell의 output 이 
        [ RNN ] <─┘  다음 cell의 input으로 연결됨
           ↑ 
           │
          x_t
```

1) 학습부분

   어떤 형태의 cell 만들지 결정 (RNN, LSTM, GRU, ...)  
   num_units 인자로 cell의 출력 크기 결정  
```
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)  
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)  
...
```

2) 구동부분  
   입력에 대한 출력값 계산, 보통 dynamic_rnn 사용  
   cell, 입력 데이터 x_data 넘겨줘  
   outputs과 state 값 나와  
   state 직접 사용은 잘 안하고, output 많이 사용  
```
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype = tf.float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Simple Example

One node

* input data  
```
dim  : 4  
one-hot encoding  
h = [1, 0, 0, 0]  
e = [0, 1, 0, 0]  
l = [0, 0, 1, 0]  
o = [0, 0, 0, 1]  

shape = (1, 1, 4) <- 여기의 4가 input dimension
(예) [[[1, 0, 0, 0]]]
```

* output data
```
hidden size = 2
shape = (1, 1, 2) <- 여기의 2 가 hidden size (출력값 개수)
[[[x, x]]]

(예) [[[0.5, 0.7]]]
```

```
           y    [[[x, x]] shape=(1, 1, 2)
           ↑
           │ ┌────┐
        [ RNN ] <─┘ hidden_size = 2
           ↑ 
           │
          x_t   [[[1, 0, 0, 0]]] shape=(1, 1, 4)
```
          


실습코드
```
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://learningtensorflow.com/index.html
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]



# 간단히 입력주고 출력 나오는 것 볼것
with tf.variable_scope('one_cell') as scope:
    # 1. 학습
    # One cell RNN input_dim (4) -> output_dim (2)
    hidden_size = 2  # 출력은 2개로
    # cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)

    # 2. 구동
    x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]]
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
```

결과
```
(2, LSTMStateTuple(c=2, h=2))
array([[[ 1.,  0.,  0.,  0.]]], dtype=float32)

weight 가 initial random value 에서 학습되서 다음과 같이 dim 2 의 값으로 나옴
array([[[-0.03028925, -0.02897934]]], dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Unfolding to N Sequences

sequence data의 길이가 shape에서 두번째 위치의 값임

input shape = (1, 5, 4) <- sequence length=5, dim=4

이렇게 지정하면 tensorflow 는 알아서  seq =5 잡아

cell 을 unfolding 해서 보면 다음처럼 생김

[  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  


```
with tf.variable_scope('two_sequances') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    
    # 입력 데이터 sequence length = 5, dim = 4
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)
    
    # dynamic_rnn에 cell, x_data 넘겨줘, 결과 outputs 로 받아
    # output의 shape = (1, 5, 2) 될 것
    outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

(1, 5, 4)
array([[[ 1.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.],
        [ 0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  1.]]], dtype=float32)
array([[[-0.65660584,  0.05588181],
        [ 0.87573957,  0.49277371],
        [-0.42493662,  0.46814129],
        [ 0.55658931,  0.30181295],
        [-0.74249279,  0.60128808]]], dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Batching Input

매번 문자열 하나하나씩 학습하면 비효율적 -> Batch 사용

```
hidden_size     = 2    결과값 2개로 표현
sequence_length = 5    문자열 길이 5
batch_size      = 3    문자열 개수 3 (예) hello, eolll, lleel
```

입력값 shape = (3, 5, 4)
```
batch 3개, sequence 5개, dim 4
[
 [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]], 
 [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]], 
 [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]], 
]
```
출력값 shape = (3, 5, 2)
```
[
    [[x, x], [x, x], [x, x], [x, x], [x, x]],
    [[x, x], [x, x], [x, x], [x, x], [x, x]],
    [[x, x], [x, x], [x, x], [x, x], [x, x]]
]
```

```
with tf.variable_scope('3_batches') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    # batch 3개, sequence 5개, dim 4
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    
    hidden_size = 2
    # 학습 (결과값 dim 2)
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    
    # 구현
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Initial State

```
with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    x_data = np.array([[h, e, l, l, o],
                      [e, o, l, l, l],
                      [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
    hidden_size=2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    # 처음 initial state 설정
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->
etc.

- bi-directional rnn

- flattern based softmax

