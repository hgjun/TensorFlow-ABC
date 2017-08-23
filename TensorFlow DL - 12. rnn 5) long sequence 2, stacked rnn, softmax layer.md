<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-5. RNN with Long Sequences: Stacked RNN + Softmax Layer



(12-4.)  
RNN 안쓰고 Softmax 사용한 경우,  
조금 긴 문자열로 RNN 해봤어

(12-5.)  
이제 진짜 긴 문자열로 RNN 할 것

<br />

참조 링크  

https://www.youtube.com/watch?v=vwjt1ZE5-K4&feature=youtu.be  
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-4-rnn_long_char.py

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Long Sequences

학습할 문장
```
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# training dataset
# 잘라서 여러개의 batch 만들어 쓸 것
# 0 if you wan -> f you want
# 1 f you want ->  you want 
# 2  you want  -> you want t
# 3 you want t -> ou want to
# …
# 168  of the se -> of the sea
# 169 of the sea -> f the sea.
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### dataset generation

일정한 sequence 크기만큼 loop 돌리며 잘라내

sequence_length = 10

```
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Parameters

```
data_dim        = len(char_set)    # dictionary 크기, one hot size로 사용되니 입력 크기가 됨
hidden_size     = len(char_set)    # RNN (cell) output size
num_classes     = len(char_set)    # final output size (RNN or softmax, etc.)
sequence_length = 10               # Any arbitrary number
learning_rate   = 0.1


# 이제 batch 한 개 이상이야
# 이 예제에서는 batch size를 dataX에 있는 문자열 전체 개수로 통으로 쓸것
batch_size      = len(dataX)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Stacked RNN

기존 1 hidden layer로는 학습 잘 안돼,
```
# cell, num_unite = hidden_size (dictionary 사이즈와 같아) 
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
```

<br />
--> MultiRNNCell 사용  
쌓을 레이어 개수만 알려주면돼!

```
cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)  # 2 layers
```

<br />
cell 생성 함수 쓰면 더 편리

```
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
```

<br />
Stacked RNN 사용한 코드

```
X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)  # 1 -> 0 1 0 0 0 0 0 0 0 0 
print(X_one_hot)  # check out the shape


# Make a lstm cell with hidden_size (출력개수)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# 정의된 multi_cells 로 static 혹은 dynamic rnn구조 정의
# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Softmax (FC) in RNN

이전 CNN 도 마지막에 FC layer 붙였어

RNN도 FC 붙여주자

그런데 sequence 개수 만큼 FC 만들어줄 필요는 없어


[ ] [ ] [ ] ... [ ]

여러개여도 사실 rnn은 하나의 rnn이야 
-> 한 개의 softmax layer 쓰도록 reshape 활용하자!


 o        o        o        o 
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])
  ↖     ↑       ↑      ↗     ==> (2) RNN 용으로 다시 reshape
       [FC: softmax]
              ↑
              o
              o
              o
              o
x_for_fc = tf.reshape(outputs, [-1, hidden_size])  ==> (1) softmax 용으로 reshape
  ↗      ↑       ↑      ↖  
 o        o        o        o 
 ↑       ↑       ↑       ↑
[  ] ->  [  ] ->  [  ] ->  [  ] -> 
 ↑       ↑       ↑       ↑ 
 x        x        x        x


(예) softmax 사용
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

# [hidden_size, num_classes]: 입력사이즈, 출력사이즈(one-hot)
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])

# softmax_outputs
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# 엄밀히 따지면 다음까지 해야하지 않을까 싶지만 
# outputs = tf.nn.softmax(outputs)

# rnn outpts
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])


==> FC 사용
# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])




<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Loss

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])


# 그냥 rnn output 을 바로 넣으면 NG
# 이제 softmax 거친 값을 logits에 주는 것이 적절한 값 전달!
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)





<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training ans print results


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
        
    # 화면에 출력
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)


# 모든 학습 끝난 후 각 배치의 출력 모아서 한번에 출력해보기
# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')



<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 전체 코드

from __future__ import print_function  # python3 print 사용 가능

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()

  
#---------------------------------------------------------------------------
### Data Preperation

# 학습할 문장
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
         
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}


#---------------------------------------------------------------------------
### Parameters

data_dim        = len(char_set)    # dictionary 크기, one hot size로 사용되니 입력 크기가 됨
hidden_size     = len(char_set)    # RNN (cell) output size
num_classes     = len(char_set)    # final output size (RNN or softmax, etc.)
sequence_length = 10               # Any arbitrary number
learning_rate   = 0.1


#---------------------------------------------------------------------------
### Dataset generation

# training dataset
# sentence 잘라 여러개의 batch 만들어 쓸 것
# 0 if you wan -> f you want
# 1 f you want ->  you want 
# 2  you want  -> you want t
# 3 you want t -> ou want to
# …
# 168  of the se -> of the sea
# 169 of the sea -> f the sea.

# 일정한 sequence_length 크기만큼 loop 돌리며 잘라내
dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

# 이제 batch 여러개임
# 이 예제에서는 batch size를 dataX에 있는 문자열 전체 개수로 통으로 쓸것
batch_size = len(dataX)


#---------------------------------------------------------------------------
### RNN Model: data to feed

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape


#---------------------------------------------------------------------------
### RNN Model: Stacked RNN

# Make a lstm cell with hidden_size (출력개수)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# 정의된 multi_cells 로 static 혹은 dynamic rnn구조 정의
# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)


#---------------------------------------------------------------------------
### RNN Model: FC layer (softmax)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# 엄밀히 따지면 다음까지 해야하지 않을까 싶지만 
# outputs = tf.nn.softmax(outputs)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])


#---------------------------------------------------------------------------
### RNN Model: min cost

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])


# 그냥 rnn output 을 바로 넣으면 NG
# 이제 softmax (FC layer) 거친 값을 logits에 주는 것이 적절한 값 전달!
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss)


#---------------------------------------------------------------------------
### Training and Results

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        _, l, results = sess.run(
            [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
        
        # 화면에 출력
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)


    # 모든 학습 끝난 후 각 배치의 출력 모아서 한번에 출력해보기
    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')


결과 (500번 loop)
0 if you wan -> f you want
1 f you want ->  you want 
2  you want  -> you want t
3 you want t -> ou want to
4 ou want to -> u want to 
5 u want to  ->  want to b
...
499 166 h of the s 0.228806
499 167  of the se 0.228806
499 168 tf the sea 0.228806
499 169   the sea. 0.228806
g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.


결과 (3000번 loop)
0 if you wan -> f you want
1 f you want ->  you want 
2  you want  -> you want t
3 you want t -> ou want to
4 ou want to -> u want to 
5 u want to  ->  want to b
...
2999 166   of the s 0.228531
2999 167  of the se 0.228531
2999 168 tf the sea 0.228531
2999 169   the sea. 0.228531
p you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.



<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 추가로 해보기

세익스피어 글 학습 RNN -> 글 생성

리눅스 코드 학습 RNN -> 소스 코드 생성 (주석, indent, ..)


더 복잡하게 구현된 것 보기
https://github.com/sherjilozair/char-rnn-tensorflow 

https://github.com/hunkim/word-rnn-tensorflow 



