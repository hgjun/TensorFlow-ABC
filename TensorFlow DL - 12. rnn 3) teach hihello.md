<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-3. RNN: teach hihello


RNN training 할 것

```

 i        h        e        l        l        o
 ↑        ↑        ↑        ↑        ↑        ↑
[  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] -> 
 ↑        ↑        ↑        ↑        ↑        ↑
 h        i        h        e        l        l 

```
같은 h 여도 어떨땐 다음 문자 i 나올 때도 e 나올때도 있어  
RNN 이전 방법으로는 문제 해결 쉽지 않아

<br />

참조 링크  
https://www.youtube.com/watch?v=39_P23TqUnw&feature=youtu.be  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-1-hello-rnn.py  

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Input Data
```
text      : hihello
vocabulary: h, i, e, l, o
vod index : h=0, i=1, e=2, l=3, o=4

one-hot encoding
[1, 0, 0, 0, 0]  h
[0, 1, 0, 0, 0]  i
[0, 0, 1, 0, 0]  e
[0, 0, 0, 1, 0]  l
[0, 0, 0, 0, 1]  o

```


```
입력 data 형태    출력 data 형태
batch    = 1       batch    = 1  # 한 개 문자열
dim      = 5       dim      = 5  # 5개로 한 문자 표현
sequence = 6       sequence = 6  # 문자열 길이 'hihell' = 6
```

```
    i           h           e           l           l           o
[0,1,0,0,0] [1,0,0,0,0] [0,0,1,0,0] [0,0,0,1,0] [0,0,0,1,0] [0,0,0,0,1]
 
    ↑           ↑           ↑           ↑           ↑           ↑
  [   ]  ->   [   ]  ->   [   ]  ->   [   ]  ->   [   ]  ->   [   ]   ->
    ↑           ↑           ↑           ↑           ↑           ↑

    h           i           h           e           l           l  
[1,0,0,0,0] [0,1,0,0,0] [1,0,0,0,0] [0,0,1,0,0] [0,0,0,1,0] [0,0,0,1,0]
 ```
 

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Creating RNN Cell

RNN, LSTM, GRU 등 사용해서 만들면 돼

출력 크기 정하는것도 중요  
출력값은 문자 표현할 one-hot encoding의 dim 이니  
rnn_size = 5 로 설정

```
# RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size) 

rnn_cell = rnn_cell.BasicLSTMCell(rnn_size)
rnn_cell = rnn_cell.BasicGRUCell(rnn_size)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Execute RNN

```
outputs, _states = tf.nn.dynamic_rnn(rnn_cell, x_data,
                   initial_state=initial_state, dtype=tf.float32)
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### RNN Parameters

```
hidden_size     = 5  # RNN cell에서 출력할 값의 크기
input_dim       = 5  # 입력값의 크기
batch_size      = 1  # 문자열 개수 (hihell 하나)
sequence_length = 6  # 문자열 길이
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Data Creation

```
# vocabulary: h, i, e, l, o
# vod index : h=0, i=1, e=2, l=3, o=4

# idx를 char로 만들어줄 dictionary 역할
idx2char = ['h', 'i', 'e', 'l', 'o']  

# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]    # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

# 학습하고 싶은 결과값 (true set)
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Feed to RNN

```
# X one-hot: shape = (1, 6, 4)
# 예제 데이터 batch size = 1 이지만, conventional 하게 None으로 했음
X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])
    
# Y label: shape(1, 6)
Y = tf.placeholder(tf.int32, [None, sequence_length])  

# Cell 제작, 출력값크기 5 (hidden_size = 5)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 구현, initial state 는 다 0 으로 설정
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, 
                   initial_state=initial_state, dtype = tf.float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost (Loss) 계산: sequence_loss

```
# 임의의 예제
# batch_size, sequence_length
y_data = tf.constant([[1, 1, 1]])

# batch_size, sequence_length, emb_dim
# output 결과는 (1, 0, 1) 인 예제
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

# batch_size, sequence_length
weights = tf.constant([[1, 1, 1,]], dtype=tf.float32)


# sequence_loss 로 lost 계산 쉽게 가능
# logits=prediction 예측한 결과 outputs
# targets=y_data    결과 정답셋
# weights=weights   문자열 각각의 자리 (dim 개수)얼마나 중요하게 생각하나, 다 1주면돼

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=prediction, targets=y_data, weights=weights)
    
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())
```

Loss: 0.596759 나와


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost (Loss) 계산: sequence_loss2 

```
# 임의의 예제
# batch_size, sequence_length
y_data = tf.constant([[1, 1, 1]])

# batch_size, sequence_length, emb_dim
# output1 결과는 (1, 0, 1), output2는 (1, 1, 1)  output2가 더 loss 낮게 나올것
prediction1 = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

# batch_size, sequence_length
weights = tf.constant([[1, 1, 1,]], dtype=tf.float32)


# sequence_loss 로 lost 계산 쉽게 가능
# logits=prediction 예측한 결과 outputs
# targets=y_data    결과 정답셋
# weights=weights   문자열 각각의 자리 얼마나 중요하게 생각하나, 다 1주면돼

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
    
sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval())
```

Loss1: 0.513015  
Loss2: 0.371101


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Cost (Loss) 계산: sequence_loss

hihello 예제
```
outputs, _states = tf.nn.dynamic_rnn(cell, X, 
                   initial_state=initial_state, dtype = tf.float32)

# (cf) 사실 outputs 값 logits로 바로 사용하면 NG (이 부분다음 섹션에서 다룰것임)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)

# loss 평균구해
loss = tf.reduce_mean(sequence_loss)

# loss 학습
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### tf.argmax


```
prediction = tf.argmax(outputs, axis=2)
```
outputs의 shape = (1, 6, 5) 이고  
axis = 2 이면  
output의 dim[2] 값 중 제일 큰 idx값 뽑겠다는 것

예제
```
test = [[
     [ 1.0, 0.3, 0.3, 0.3, 0.3 ],
     [ 0.3, 1.0, 0.3, 0.3, 0.3 ],
     [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
     [ 0.3, 0.3, 1.0, 0.3, 0.3 ],
     [ 0.3, 0.3, 0.3, 1.0, 0.3 ]
  ]]
  
with tf.Session() as sess:
    print(sess.run(tf.argmax(test, axis=2)))
```
결과
```
[[0 1 2 2 3]]
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training

```
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
    
	    # x, y 데이터로 train 학습시켜
	    # (정보 print위해 loss 도 run)
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        # dictionary 이용해 one-hot 에서 문자열로 직접 보기 
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 전체 코드

```
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

# Teach hello: hihell -> ihello

#---------------------------------------------------------------------------
### Data Creation

# vocabulary: h,   i,   e,   l,   o
# voca index: h=0, i=1, e=2, l=3, o=4

# idx를 char로 만들어줄 dictionary 역할
idx2char = ['h', 'i', 'e', 'l', 'o']  

# Teach hello: hihell -> ihello
x_data    = [[0, 1, 0, 2, 3, 3]] # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

# 학습하고 싶은 결과값 (target value)
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello


#---------------------------------------------------------------------------
### Parameters

num_classes     = 5   # 5자리로 one-hot encoding되어있음 
input_dim       = 5   # one-hot size
hidden_size     = 5   # output from cell. 5 to directly predict one-hot
batch_size      = 1   # one sentence
sequence_length = 6   # 문자열 길이|ihello| == 6
learning_rate   = 0.1


#---------------------------------------------------------------------------
### Feed to RNN

# X one-hot: shape = (1, 6, 4)
# 예제 데이터 batch size = 1 이지만, conventional 하게 None으로 했음
X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])
    
# Y label: shape(1, 6)
Y = tf.placeholder(tf.int32, [None, sequence_length])  

# Cell, 출력값크기 5 (hidden_size = 5)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 구현, initial state 는 다 0 으로 설정
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, 
                   initial_state=initial_state, dtype = tf.float32)

# print(outputs)
# Tensor("rnn/transpose:0", shape=(1, 6, 5), dtype=float32)


#---------------------------------------------------------------------------
### FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# fc layer 거친 후 outputs
# print(outputs)
# Tensor("fully_connected/BiasAdd:0", shape=(6, 5), dtype=float32)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# loss 계산위해 reshape 이용해 원래 outputs의 shape 복원
# print(outputs)
# Tensor("Reshape_1:0", shape=(1, 6, 5), dtype=float32)


#---------------------------------------------------------------------------
### Cost (Loss) 계산: sequence_loss

# logits=prediction 예측한 결과 outputs
# targets=y_data    결과 정답셋
# weights=weights   문자열 각각의 자리 얼마나 중요하게 생각하나, 다 1주면돼

weights = tf.ones([batch_size, sequence_length])  # [1, 6]크기로 1채워
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)   

# loss 평균구해
loss = tf.reduce_mean(sequence_loss)

# loss 학습
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#---------------------------------------------------------------------------
### Set prediction from outputs

# outputs의 shape = (1, 6, 5)
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
### Training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(50):    
	    # x, y 데이터로 train 학습시켜
	    # (정보 확인위해 loss 도 print)
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        # dictionary 이용해 one-hot 에서 문자열로 직접 보기 
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
```

실행 결과
```
0, 'loss:', 1.6215916, 'prediction: ', array([[3, 3, 3, 3, 3, 3]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'llllll')
(1, 'loss:', 1.5033671, 'prediction: ', array([[3, 3, 3, 3, 3, 3]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'llllll')
(2, 'loss:', 1.4078447, 'prediction: ', array([[3, 3, 3, 3, 3, 3]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'llllll')
(3, 'loss:', 1.3134491, 'prediction: ', array([[3, 3, 3, 3, 3, 3]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'llllll')
(4, 'loss:', 1.2051442, 'prediction: ', array([[3, 3, 3, 3, 3, 3]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'llllll')
(5, 'loss:', 1.0661778, 'prediction: ', array([[0, 0, 0, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'hhhllo')
...

(45, 'loss:', 0.0073211561, 'prediction: ', array([[1, 0, 2, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'ihello')
(46, 'loss:', 0.0068840194, 'prediction: ', array([[1, 0, 2, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'ihello')
(47, 'loss:', 0.006490354, 'prediction: ', array([[1, 0, 2, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'ihello')
(48, 'loss:', 0.0061377473, 'prediction: ', array([[1, 0, 2, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'ihello')
(49, 'loss:', 0.0058230362, 'prediction: ', array([[1, 0, 2, 3, 3, 4]]), 'true Y: ', [[1, 0, 2, 3, 3, 4]])
('\tPrediction str: ', 'ihello')
```

<br />

참조 링크  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-1-hello-rnn.py


<br /><br />


