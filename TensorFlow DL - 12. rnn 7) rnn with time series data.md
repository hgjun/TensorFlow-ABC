<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-7. RNN with Time Series Data


Time Series Data(주식 등) 예측하는 RNN 해볼 것

x축: time  
y축: value

<br />
참조 링크  

https://www.youtube.com/watch?v=odMGK7pwTqY&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### RNN: many to one

RNN의 구조 many to one 으로 하는 idea
```
                                                       8
                                                       ↑
[  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ]
 ↑        ↑        ↑        ↑        ↑        ↑        ↑
 1        2        3        4        5        6        7

```
Time series 의 가설에 의해  
8일째 값 예측할때,  
하루전 data 하나만 있는것보다, 몇일 전부터의 여러 data 있는게 더 잘될 것


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Example: Stock data

|      |  x1  |  x2  |  x3 |  x4  |   y   |
|:----:|:----:|:----:|:---:|:----:|:-----:|
|      | open | high | low |  vol | close |
| day1 |  800 |  850 | 790 | 1000 |  830  |
| day2 |  825 |  860 | 810 | 1005 |  840  |
| day3 |  830 |  870 | 820 | 1010 |  845  |
|  ... |  ... |  ... | ... |  ... |  ...  |
| day8 |  835 |  875 | 830 | 1015 |   ?   |

<br />

day8 의 close 값이 알고 싶다면,

```
                                                     day8
                                                       ↑
[  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ]
 ↑        ↑        ↑        ↑        ↑        ↑        ↑
day1     day2     day3     day4     day5     day6     day7
```
```
data_dim        = 5  # input dim: cell에 open, high, low, vol, close 5개 값을 줘
sequence_length = 7  # seq 길이: 8일차 예측 위해 7일간 데이터를 7개 cell에 줘
output_dim      = 1  # 8일차의 close 값을 구하므로
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Reading Data

우선 작은 데이터 사용 (data04_stock_daily_small.csv)

```
# small data
1,2,3,4,10
2,3,4,5,11
3,4,5,6,12
4,4,4,4,13
5,5,5,5,17
6,6,6,6,18
2,3,2,3,20
1,1,2,2,31
2,2,2,4,40
2,1,4,5,50
```

```
import tensorflow as tf
import random
import os
os.chdir('/home/testu/work') 
os.getcwd()

import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility
    
# train Parameters
seq_length    = 7     # seq_length = timestamps
data_dim      = 5     # input dim: open, high, low, vol, close
hidden_dim    = 10
output_dim    = 1
learning_rate = 0.01
iterations    = 500

# data generation
xy = np.loadtxt('data04_stock_daily_small.csv', delimiter=',')
xy = xy[::-1]         # 시간순으로 만들기위해 한번 뒤집어 reverse order (chronically ordered)
# xy = MinMaxScaler(xy) # 값이 편차크니 normalize 시킴
x = xy                # open, high, low, vol, close  5개 전체
y = xy[:, [-1]]       # label = 'close' 값만


print(x)
print(y)
          
# dataset generation
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]   # i   일
    _y = y[i + seq_length]     # i+1 일의 y값 (다음날의 정답값)
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
```

<br />
x 값 (뒤집었으니 csv 맨 밑에 값이  처음 값이 됨)

```
       x 값 (dim = 5)                원본 csv
[[  2.   1.   4.   5.  50.]        [1,2,3,4,10]
 [  2.   2.   2.   4.  40.]        [2,3,4,5,11]
 [  1.   1.   2.   2.  31.]        [3,4,5,6,12]
 [  2.   3.   2.   3.  20.]        [4,4,4,4,13]
 [  6.   6.   6.   6.  18.]        [5,5,5,5,17]
 [  5.   5.   5.   5.  17.]        [6,6,6,6,18]
 [  4.   4.   4.   4.  13.]        [2,3,2,3,20]
 [  3.   4.   5.   6.  12.]        [1,1,2,2,31]
 [  2.   3.   4.   5.  11.]        [2,2,2,4,40]
 [  1.   2.   3.   4.  10.]]       [2,1,4,5,50]
```

<br />
y값 (뒤집었으니 csv 맨 밑에 값이  idx=0 값이 됨)

```
[[ 50.]
 [ 40.]
 [ 31.]
 [ 20.]
 [ 18.]
 [ 17.]
 [ 13.]
 [ 12.]
 [ 11.]
 [ 10.]]
```

<br />
data 10개 - seg_length 7 = 3 번 반복가능  
x dim = 5, sequence = 7, y dim = 1

```
(array([[  2.,   1.,   4.,   5.,  50.],
       [  2.,   2.,   2.,   4.,  40.],
       [  1.,   1.,   2.,   2.,  31.],
       [  2.,   3.,   2.,   3.,  20.],
       [  6.,   6.,   6.,   6.,  18.],
       [  5.,   5.,   5.,   5.,  17.],
       [  4.,   4.,   4.,   4.,  13.]]), '->', array([ 12.]))
(array([[  2.,   2.,   2.,   4.,  40.],
       [  1.,   1.,   2.,   2.,  31.],
       [  2.,   3.,   2.,   3.,  20.],
       [  6.,   6.,   6.,   6.,  18.],
       [  5.,   5.,   5.,   5.,  17.],
       [  4.,   4.,   4.,   4.,  13.],
       [  3.,   4.,   5.,   6.,  12.]]), '->', array([ 11.]))
(array([[  1.,   1.,   2.,   2.,  31.],
       [  2.,   3.,   2.,   3.,  20.],
       [  6.,   6.,   6.,   6.,  18.],
       [  5.,   5.,   5.,   5.,  17.],
       [  4.,   4.,   4.,   4.,  13.],
       [  3.,   4.,   5.,   6.,  12.],
       [  2.,   3.,   4.,   5.,  11.]]), '->', array([ 10.]))
       
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training and Test Datasets

데이터 나눠: 70% tr data, 30% test data
```
# train/test split
train_size = int(len(dataY) * 0.7)
test_size  = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), 
                np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), 
                np.array(dataY[train_size:len(dataY)])
```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### LSTM + FC and Loss
```
                                             y_pred
                                              ↑
                                           [FC layer] 하나 더 둬!
                                              ↑
[  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ] ->  [  ]
 ↑        ↑        ↑        ↑        ↑        ↑
day1     day2     day3     day4     day5     day6
```

FC_layer output size = 1 이면 되니  
cell output의 크기 num_units = hidden_dim 은 임의로 정할 수 있어!

```
# input place holders
# None for batch_size
X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) # (?, 7, 5)
Y = tf.placeholder(tf.float32, [None, 1])                    # (?, 1)

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# FC layer
# many to one (We use the last cell's output)
# seq_length = 7 만큼 있는 cell 7 개 중에
# 맨 마지막에 있는 cell의 output만 input으로 사용하겠다!
# (예) 8일차 y 예측은 7일차 통해서 예측
# 입력 outputs[:, -1]
# 출력 output_dim = 1
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None) 

# cost/loss
# 하나만 보니 sequence loss 아닌 linear losss 이므로 mean squared err 사용
loss = tf.reduce_mean(tf.square(Y_pred - Y))  # mean squared error
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training and Results

```
# RMSE (root mean squared error)
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], 
                       feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

```
뭐가 true 값이고 뭐가 예측값인지 구분못할정도로 결과 비슷하게 나와  
-> RNN으로 Sequential Data 분석 잘된다


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 전체 코드

```
import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

os.chdir('/home/testu/work') 
os.getcwd()

tf.set_random_seed(777)  # for reproducibility

# 이전 실험한 코드의 그래프 variables 가 남아있으면 
# 지금 코드의 variable 과 이름 같은 경우 에러나므로
# 우선 reset 시키고 시작
tf.reset_default_graph()

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

#---------------------------------------------------------------------------
### MinMaxScaler function for input data normalization

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)
    
#---------------------------------------------------------------------------
### Parameters
seq_length    = 7     # seq_length = timestamps
data_dim      = 5     # input dim: open, high, low, vol, close
hidden_dim    = 10
output_dim    = 1
learning_rate = 0.01
iterations    = 500


#---------------------------------------------------------------------------
### Data Preperation

# data generation
xy = np.loadtxt('data04_stock_daily.csv', delimiter=',')
xy = xy[::-1]         # 시간순으로 만들기위해 한번 뒤집어 reverse order (chronically ordered)
xy = MinMaxScaler(xy) # 값이 편차크니 normalize 시킴
x = xy                # open, high, low, vol, close  5개 전체
y = xy[:, [-1]]       # label = 'close' 값만
          
# dataset generation
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]   # i   일
    _y = y[i + seq_length]     # i+1 일의 y값 (다음날의 정답값)
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
    

#---------------------------------------------------------------------------
### Training and test datasets

# train/test split: 70% / 30%
train_size = int(len(dataY) * 0.7)
test_size  = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


#---------------------------------------------------------------------------
### RNN Model: data to feed

# input place holders
# None for batch_size
X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) # (?, 7, 5)
Y = tf.placeholder(tf.float32, [None, 1])                    # (?, 1)


#---------------------------------------------------------------------------
### RNN Model: LSTM network

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


#---------------------------------------------------------------------------
### RNN Model: FC layer

# many to one (We use the last cell's output)
# seq_length = 7 만큼 있는 cell 7 개 중에
# 맨 마지막에 있는 cell의 output만 input으로 사용하겠다!
# (예) 8일차 y 예측은 7일차 통해서 예측
# 입력 outputs[:, -1]     출력 output_dim = 1
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None) 


#---------------------------------------------------------------------------
### RNN Model: min cost

# 하나만 보니 sequence loss 아닌 linear losss 이므로 mean squared err 사용
loss = tf.reduce_mean(tf.square(Y_pred - Y))  # mean squared error

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train     = optimizer.minimize(loss)


#---------------------------------------------------------------------------
### Training and Results

# RMSE (root mean squared error)
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], 
                       feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
    # 뭐가 true 값이고 뭐가 예측값인지 구분못할정도로 비슷하게 나와
```

결과
```
[step: 0] loss: 0.27139043808
[step: 1] loss: 0.155455857515
[step: 2] loss: 0.0739140063524
[step: 3] loss: 0.0311691779643
[step: 4] loss: 0.0281070284545
[step: 5] loss: 0.0484252758324
...
[step: 495] loss: 0.00100397854112
[step: 496] loss: 0.00100330437999
[step: 497] loss: 0.00100263324566
[step: 498] loss: 0.00100196502171
[step: 499] loss: 0.00100129982457
RMSE: 0.0250787623227
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Exercises

Stock Prediction을 Linear Regression 만 가지고 해보기  
그래서 RNN (LSTM)과 얼마나 차이 나는지 봐봐  

여러 다른 Feature 들 사용해서 더 잘 예측해보기  
(키워드, 관련기사, sentiments in top news, 등등)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Other RNN Applications

RNN은 재미있고, 활용할 것도 많은 분야

- Language Modeling
- Speech Recognition
- Machine Translation
- Conversation Modeling/Question Answering
- Image/Video Captioning
- Image/Music/Dance Generation


