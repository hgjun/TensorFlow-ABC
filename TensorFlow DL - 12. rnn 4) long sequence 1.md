<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-4. RNN with long sequences


<br />

참조 링크  
https://www.youtube.com/watch?v=2R6nfCNNz1U&feature=youtu.be

1. NN 안쓰고 softmax 사용한 경우  
http://192.168.99.100:8888/notebooks/Untitled2.ipynb?kernel_name=python2


2. 긴 문자열 RNN  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-2-char-seq-rnn.py



<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1. RNN 안쓰고 Softmax 사용한 경우

```
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# flatten the data (ignore batches for now). No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

# Compute sequence cost/loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))
```

결과 좋지 않음
```
(0, 'loss:', 2.5366843, 'Prediction:', 'anaaooaannwaaoo')
(1, 'loss:', 2.3162146, 'Prediction:', 'wy wooawnnwowoo')
(2, 'loss:', 2.1114612, 'Prediction:', 'wy wooywanyowoo')
(3, 'loss:', 1.9229985, 'Prediction:', 'yy yooyyanyoyoo')
...
(2997, 'loss:', 0.27732366, 'Prediction:', 'yf you yant you')
(2998, 'loss:', 0.27732363, 'Prediction:', 'yf you yant you')
(2999, 'loss:', 0.2773236, 'Prediction:', 'yf you yant you')
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 2. 긴 문자열

#### Manual Data Creation

지난번 방식 -> 긴 문자열이면 매번 이렇게 입력하기 힘들어!
```
idx2char = ['h', 'i', 'e', 'l', 'o']  

x_data    = [[0, 1, 0, 2, 3, 3]] # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

# 학습하고 싶은 결과값 (target value)
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Better Data Creation

```
sample   = " if you want you"

# index -> char
# ['a', ' ', 'f', 'i', 'o', 'n', 'u', 't', 'w', 'y']
idx2char = list(set(sample))

# char -> index
# {' ': 1, 'a': 0, 'f': 2, 'i': 3, 'n': 5, 'o': 4, 't': 7, 'u': 6, 'w': 8, 'y': 9}
char2idx = {c: i for i, c in enumerate(idx2char)}

# sample data의 char to index
# [1, 3, 2, 1, 9, 4, 6, 1, 8, 0, 5, 7, 1, 9, 4, 6]
sample_idx = [char2idx[c] for c in sample]  

x_data = [sample_idx[:-1]]  # X data  sample (0 ~ n-1) hello -> hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n)   hello -> ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# tf.one_hot 으로 쉽게 one hot encoding 가능
# 주의 tf.one_hot 쓰면 dim 좀 차이 있으니 shape 체크해봐야
# num_classes = len(char2idx)  최종 출력값 크기
x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

# print(x_one_hot)
# Tensor("one_hot_2:0", shape=(?, 6, 10), dtype=float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Parameters

```
dic_size        = len(char2idx)    # dictionary 크기, one hot size로 사용되니 입력 크기가 됨
hidden_size     = len(char2idx)    # RNN (cell) output size
num_classes     = len(char2idx)    # final output size (RNN or softmax, etc.)
batch_size      = 1                # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #) (ex) hello -> hell
learning_rate   = 0.1
```


<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### LSTM and Loss

```
# cell, num_unite = hidden_size (dictionary 사이즈와 같아) 
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 구현, initial state 는 다 0 으로 설정
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)


# sequence_loss 로 lost 계산
# logits=outputs    예측한 결과 outputs
# targets=Y         결과 정답셋
# weights=weights   문자열 각각의 자리 (dim 개수)얼마나 중요하게 생각하나, 다 1주면돼
weights = tf.ones([batch_size, sequence_length]) # [1, 10]
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)

# loss 평균내서 학습
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# axis = 2 이므로 dim[2] 값 중 젤 큰 idx값 뽑겠다는 것
# outputs = [[[ 1.0, 0.3, 0.3, 0.3, 0.3 ],
#             [ 0.3, 1.0, 0.3, 0.3, 0.3 ],
#             [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
#             [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
#             [ 0.3, 0.3, 0.3, 1.0, 0.3 ]]]
# 이면,
# prediction = [[0, 1, 2, 2, 3]] 될 것
prediction = tf.argmax(outputs, axis=2)
```

<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Numpy: squeeze

Remove single-dimensional entries from the shape of an array.

- Parameters
  - a   : array_like Input data.
  - axis: None or int or tuple of ints, optional
    - Selects a subset of the single-dimensional entries in the shape.
    - If an axis is selected with shape entry greater than one, an error is raised.
       
- Returns
  - squeezed : ndarray
    - The input array, but with all or a subset of the dimensions of length 1 removed.
    - This is always a itself or a view into a.

- Raises
  - ValueError: If axis is not None, and an axis being squeezed is not of length 1

(예)
```
x = np.array([
                [ [0], [1], [2]  ]
             ])
     
x.shape
# (1, 3, 1)

np.squeeze(x).shape
# (3,)

np.squeeze(x, axis=0).shape
# (3, 1)

np.squeeze(x, axis=1).shape
# Traceback (most recent call last):
...
ValueError: cannot select an axis to squeeze out which has size not equal to one

np.squeeze(x, axis=2).shape
# (1, 3)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### printing results (테스트용)

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dictionary
        # result 프린트해볼것
        print("result", result)
        print("shape", result.shape)
        
        # squeeze 는 차원 [ ] 하나 벗겨내 (reshape 사용도 가능)
        print("squeeze", np.squeeze(result))
        print("reshape", sess.run(tf.reshape(result, [-1])))
        
        result_str = [idx2char[c] for c in np.squeeze(result)] # [1, 15] - > [15] by squeeze

        print(i, "loss:", l, "Prediction:", ''.join(result_str))
```
```
('result', array([[9, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1]]))
('shape', (1, 15))
('squeeze', array([9, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1]))
('reshape', array([9, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1]))
('squeeze shape', (15,))
(0, 'loss:', 2.3132071, 'Prediction:', 'y   o        o ')
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

#### Training and Results

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 문자열 길어서 1000번 정도로는 안돼
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dictionary
        # result 프린트해볼것
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 전체코드

```
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()

# Teach hello: " if you want yo" -> "if you want you"


#---------------------------------------------------------------------------
### Data Preperation

sample   = " if you want you"

# idx를 char로 만들어줄 dictionary 역할
# index -> char
# ['a', ' ', 'f', 'i', 'o', 'n', 'u', 't', 'w', 'y']
idx2char = list(set(sample))

# char -> index
# {' ': 1, 'a': 0, 'f': 2, 'i': 3, 'n': 5, 'o': 4, 't': 7, 'u': 6, 'w': 8, 'y': 9}
char2idx = {c: i for i, c in enumerate(idx2char)}

# sample data의 char to index
# [1, 3, 2, 1, 9, 4, 6, 1, 8, 0, 5, 7, 1, 9, 4, 6]
sample_idx = [char2idx[c] for c in sample]  

x_data = [sample_idx[:-1]]  # X data  sample (0 ~ n-1) hello -> hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n)   hello -> ello


#---------------------------------------------------------------------------
### Parameters

dic_size        = len(char2idx)    # dictionary 크기, one hot size로 사용되니 입력 크기가 됨
hidden_size     = len(char2idx)    # RNN (cell) output size
num_classes     = len(char2idx)    # final output size (RNN or softmax, etc.)
batch_size      = 1                # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #) (ex) hello -> hell
learning_rate   = 0.1


#---------------------------------------------------------------------------
### RNN Model: data to feed

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data  " if you want yo"
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label "if you want you"

# tf.one_hot 으로 쉽게 one hot encoding 가능
# 주의 tf.one_hot 쓰면 dim 좀 차이 있으니 shape 체크해봐야
# num_classes = len(char2idx)  최종 출력값 크기
x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

# print(x_one_hot)
# Tensor("one_hot:0", shape=(?, 15, 10), dtype=float32)


#---------------------------------------------------------------------------
### RNN Model: LSTM

# cell, num_unite = hidden_size (dictionary 사이즈와 같아) 
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 구현, initial state 는 다 0 으로 설정
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# print(outputs)
# Tensor("rnn/transpose:0", shape=(1, 15, 10), dtype=float32)

#---------------------------------------------------------------------------
### RNN Model: FC layer

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# fc layer 거친 후 outputs
# print(outputs)
# Tensor("fully_connected/BiasAdd:0", shape=(15, 10), dtype=float32)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# loss 계산위해 reshape 이용해 원래 outputs의 shape 복원
# print(outputs)
# Tensor("Reshape_1:0", shape=(1, 15, 10), dtype=float32)


#---------------------------------------------------------------------------
### RNN Model: min cost

# sequence_loss 로 lost 계산
# logits=outputs    예측한 결과 outputs
# targets=Y         결과 정답셋
# weights=weights   문자열 각각의 자리 (dim 개수)얼마나 중요하게 생각하나, 다 1주면돼
weights = tf.ones([batch_size, sequence_length]) # [1, 10]
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)

# loss 평균내서 학습
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#---------------------------------------------------------------------------
### Set prediction from outputs

# axis = 2 이므로 dim[2] 값 중 젤 큰 idx값 뽑겠다는 것
# outputs = [[[ 1.0, 0.3, 0.3, 0.3, 0.3 ],
#             [ 0.3, 1.0, 0.3, 0.3, 0.3 ],
#             [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
#             [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
#             [ 0.3, 0.3, 0.3, 1.0, 0.3 ]]]
# 이면,
# prediction = [[0, 1, 2, 2, 3]] 될 것
prediction = tf.argmax(outputs, axis=2)


#---------------------------------------------------------------------------
### Training and Results

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 문자열 길어서 1000번 정도로는 안돼
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dictionary
        # result 프린트해볼것
        # result  = [[9, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1]], shape (1, 15)
        # squeeze -> [9, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1] , shape (15)
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))
```

결과
```
(0, 'loss:', 2.3123939, 'Prediction:', 'yyyyooooyoyyyoo')
(1, 'loss:', 2.1762667, 'Prediction:', 'y              ')
(2, 'loss:', 2.0476766, 'Prediction:', 'y              ')
(3, 'loss:', 1.9016597, 'Prediction:', 'y  you    t  ou')
(4, 'loss:', 1.6816961, 'Prediction:', 'yy you  ynt you')
(5, 'loss:', 1.4104124, 'Prediction:', 'yyyyou yynt you')
...
(2996, 'loss:', 3.7193215e-06, 'Prediction:', 'if you want you')
(2997, 'loss:', 3.7193215e-06, 'Prediction:', 'if you want you')
(2998, 'loss:', 3.7193215e-06, 'Prediction:', 'if you want you')
(2999, 'loss:', 3.7193215e-06, 'Prediction:', 'if you want you')
```



