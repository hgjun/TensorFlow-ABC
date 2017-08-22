<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 11-2. CNN TensorFlow

CNN: 이미지, 텍스트 분야에서 좋은 성능 보임  
TensorFlow 로 CNN 구현할 것

크게 세 부분

1 입력

2 Feature Extraction  
  - convolution - subsampling (filter 여러개쓰면서 데이터 커지니 데이터 작게 만들어줘)

3 Classification  
  - fully connected layers (일반적인 forward neural net)

<br />
(활용 예)  
CNN for CT images  
Asan Medical Center & Microsoft Medical Bigdata Contest Winner by GeunYoung Lee and Alex Kim 
https://www.slideshare.net/GYLee3/ss-72966495


<br />
참조 링크  
https://www.youtube.com/watch?v=E9Xh_fc9KnQ&feature=youtu.be
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-0-cnn_basics.ipynb


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### CNN

#### Fully Connected Layer 의 문제점 (CNN 이전)

데이터의 형상이 무시된다

(예)  
256 x 256 이미지 데이터 n개  
행렬로 바꾸면  
column size: 256 x 256 = 65,536  
row size: n  
여기에 W1, W2 ... 등을 곱해 마지막에 Classification 하는 Network Layer

그러나 이미지의 픽셀은 좌우, 상하의 데이터와도 관계가 있는데 이 정보 사용 못함

<br />
#### CNN Layer

기본 구조  
중간: Convolution layer, max pooling layer, activation layer 구축  
최종: Fully connected layer 를 통해 Classification

(이외 많은 다양한 CNN Layer 구성 방법 있음)

 
<br />
#### Convolution Operation

filter 가 Image 위를 stride 크기만큼씩 이동하면서 계산하는 방법이
convolution operation (합성곱 연산)

https://en.wikipedia.org/wiki/Convolutional_neural_network

input image -> convolutional layer (convolution operation) -> ...

(뒤에 예제 있음)


* Padding, Stride

합성곱 연산에서 문제점이 점점 데이터가 소실 된다는 것이다. 
그래서 해결책으로 패딩이란것이 제시되었는데 데이터의 사이즈가 
줄어들지 않도록 가장자리에 가상의 값인 0으로 채워서 
convolutional layer 를 지나도 사이즈가 줄어들지 않도록 하는 기법


* Pooling

pooling 에는 max pooling, min pooling, avg pooling 등이 있다
CNN 에서 대표적으로 가장 많이 쓰이는 pooling 은 max pooling

참조 링크  
http://ggoals.tistory.com/87


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 테스트용 간단한 이미지

Simple Convolution Layer

이미지  
3 x 3 x 1 image (1은 색깔: 검은색 하나만 사용)

필터  
2 x 2 x 1, stride 1

출력  
2 x 2 x 1  (공식의해, (3 - 2)/1 + 1 = 2)  
TensorFlow 는 알아서 계산해줘

```
%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

# (3 x 3 x 1)      3 x 3 크기, 1 color 
# (1 x 3 x 3 x 1)  1개의 이미지 인스턴스 사용하겠다 n = 1
image = np.array([[
                   [[1],[2],[3]],
                   [[4],[5],[6]], 
                   [[7],[8],[9]]
                 ]], dtype=np.float32)
                 
print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Simple Convolution Layer

#### Image  

(batch, h, w, color) 1, 3, 3, 1  
batch: 데이터개수


| 1 | 2 | 3 |
|---|---|---|
| 4 | 5 | 6 |
| 7 | 8 | 9 |


#### Filter  

(h, w, color, n) 2, 2, 1, 1  
2 x 2 크기, 1 색깔, 1 개 필터  
Stride : 1 x 1  
Padding: VALID or SAME

| 1 | 1 |
|---|---|
| 1 | 1 |


#### Convolution layer 에서 실행되는 Convolution operation 과정
```
1)
┌ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─┐
│ 1   2   -  │ x │ 1   1  │ = 1*1 + 2*1 + 4*1 + 5*1 = 12   │12      │
│ 4   5   -  │   │ 1   1  │                                │        │
│ -   -   -  │   └ ─ ─ ─ ─┘                                └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

2)
┌ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─┐
│ -   2   3  │ x │ 1   1  │ = 2*1 + 3*1 + 5*1 + 6*1 = 16   │ 12  16 │
│ -   5   6  │   │ 1   1  │                                │        │
│ -   -   -  │   └ ─ ─ ─ ─┘                                └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

3)
┌ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─┐
│ -   -   -  │ x │ 1   1  │ = 4*1 + 5*1 + 7*1 + 8*1 = 16   │ 12  16 │
│ 4   5   -  │   │ 1   1  │                                │ 24     │
│ 7   8   -  │   └ ─ ─ ─ ─┘                                └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

3)
┌ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─┐
│ -   -   -  │ x │ 1   1  │ = 5*1 + 6*1 + 8*1 + 9*1 = 28   │ 12  16 │
│ -   5   6  │   │ 1   1  │                                │ 24  28 │
│ -   8   9  │   └ ─ ─ ─ ─┘                                └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘
```

#### Output

| 12 | 16 |
|----|----|
| 24 | 28 |


==> tf.nn.conv2d( ) 사용하면 됨!

```
tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')

strides=[1, 1, 1, 1]  # stride = (1 x 1)
첫 번째, 네 번째 인덱스는 반드시 1 사용
두 번째, 세 번째 인덱스 값이 stride 값, 보통 둘 다 같은 값 사용


# Image
print("image.shape", image.shape)

# Filter
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)


# stride = (1 x 1)
# 첫 번째, 네 번째 인덱스는 반드시 1 사용
# 두 번째, 세 번째 인덱스 값이 stride 값, 보통 둘 다 같은 값 사용
# stride (_, 1, 1, _)이 값이 1 x 1 의미
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID') # stride (_, 1, 1, _)이 값이 1 x 1 의미
conv2d_img = conv2d.eval()

# Output
print("conv2d_img.shape", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)  # 출력값 시각화
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
```

실행 결과
```
('image.shape', (1, 3, 3, 1))
('weight.shape', TensorShape([Dimension(2), Dimension(2), Dimension(1), Dimension(1)]))
('conv2d_img.shape', (1, 2, 2, 1))
[[ 12.  16.]
 [ 24.  28.]]
```
 
<br />
참조 링크  
http://goodtogreate.tistory.com/entry/Convolutional-Neural-Network-CNN-%EA%B5%AC%ED%98%84


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### TensorFLow: conv2d

```
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
```

- input
  - 4-D tensor, shape = [batch, in_height, in_width, in_channels]    
  - batch : 데이터 갯수
  - in_height : row 수
  - in_width : col 수
  - in_channels : 채널 수
  - (예) 1, 7, 7, 3   (7 x 7 x 3) 의 3색상 이미지
	
- filter : 4-D tensor, shape = [filter_height, filter_width, in_channels, out_channels]
  - filter_height : filter row 수
  - filter_width : filter col 수
  - in_channels : filter channel 수
  - out_channels : output channels 수 
  - (예) 4, 4, 3, 6   (4 x 4 x 3) 의 3색상 필터가 6개

- strides
  - 1-D tensor, shape = [ 1, ?, ?, 1 ]
  - 첫 번째, 네 번째 인덱스는 반드시 1 사용
  - 두 번째, 세 번째 인덱스 값이 stride 값, 보통 둘 다 같은 값 사용

- padding
  - "VAILD", "SAME" 
  - VALID: 패딩 사용X, 데이터 사이즈 줄어듬
  - SAME : 패딩 사용O. 입력 데이터 사이즈가 유지
	
- use_cudnn_on_gpu
  - gpu 사용 여부 ( default : true )

- data_format
  - 입력, 출력 데이터의 포멧 ( default : "NHWC" )
  - NHWC [batch, height, width, channels]
  - NCHW [batch, channels, height, width]
	
- name
  - conv2d operation name 지정 (optional)

<br />
참조 링크
https://www.tensorflow.org/api_docs/python/tf/nn/conv2d


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Padding

(예)
- Image = [1, 3, 3, 1]
  - (n, w, h, channel)

- Filter = [2, 2, 1, 1]
  - (w, h, channel, n)
  - 2 x 2 크기, 1 색깔(channel/color), 1 개 필터


#### Padding = SAME
  - Filter 크기 상관 없이 원래 이미지 사이즈와 같게 출력
  - TensorFlow 가 알아서 0으로 Padding 시켜줌 (stride 1 x 1 기준)
  - input (3 x 3 x 1) 이면 output 도 (3 x 3 x 1) 
  - stride = 2 이면 원래 이미지의 반으로 줄여 출력
 
#### Padding = VALID
  - 패딩 사용안함
  - input (3 x 3 x 1), filter (2 x 2), stride 1 -> output (2 x 2 x 1)
  - 공식의해, (input size 3 - filter size 2)/ (stride 1) + 1 = 2


#### Padding with zero
```
   input image                            after padding
  ┌ ─ ─ ─ ─ ─ ─┐                         ┌ ─ ─ ─ ─ ─ ─ ─ ─┐
  │ 1   2   3  │                         │ 1   2   3   0  │
  │ 4   5   6  │         =====>          │ 4   5   6   0  │
  │ 7   8   9  │                         │ 7   8   9   0  │
  └ ─ ─ ─ ─ ─ ─┘                         │ 0   0   0   0  │ 
                                         └ ─ ─ ─ ─ ─ ─ ─ ─┘
```

### Convolution steps
```
1)
┌ ─ ─ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─ ─ ─┐
│ 1   2   -   0  │ x │ 1   1  │ = 1*1 + 2*1 + 4*1 + 5*1 = 12   │12          │
│ 4   5   -   0  │   │ 1   1  │                                │            │
│ -   -   -   0  │   └ ─ ─ ─ ─┘                                │            │
│ 0   0   0   0  │                                             └ ─ ─ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─ ─ ─┘

2)
┌ ─ ─ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─ ─ ─┐
│ -   2   3   0  │ x │ 1   1  │ = 2*1 + 3*1 + 5*1 + 6*1 = 16   │ 12  16     │
│ -   5   6   0  │   │ 1   1  │                                │            │
│ -   -   -   0  │   └ ─ ─ ─ ─┘                                │            │
│ 0   0   0   0  │                                             └ ─ ─ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─ ─ ─┘

3)
┌ ─ ─ ─ ─ ─ ─ ─ ─┐   ┌ ─ ─ ─ ─┐                                ┌ ─ ─ ─ ─ ─ ─┐
│ -   -   3   0  │ x │ 1   1  │ = 3*1 + 0*1 + 6*1 + 0*1 =  9   │ 12  16  9  │
│ -   -   6   0  │   │ 1   1  │                                │            │
│ -   -   -   0  │   └ ─ ─ ─ ─┘                                │            │
│ 0   0   0   0  │                                             └ ─ ─ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─ ─ ─┘

 ....

```

```

# Image
print("image.shape", image.shape)

# Filter
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME') # padding='SAME': 입력크기 = 출력크기
conv2d_img = conv2d.eval()


# 여러 개 변수 쓰면 헷갈릴 수 있어, 
# Shape 헷갈리면 Print tensor 해봐
print(conv2d)  # Tensor("Conv2D_14:0", shape=(1, 3, 3, 1), dtype=float32)



# Output
print("conv2d_img.shape", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3) # 출력값 시각화
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    

('image.shape', (1, 3, 3, 1))
('weight.shape', TensorShape([Dimension(2), Dimension(2), Dimension(1), Dimension(1)]))
('conv2d_img.shape', (1, 3, 3, 1))
[[ 12.  16.   9.]
 [ 24.  28.  15.]
 [ 15.  17.   9.]]
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 여러 Filter 사용

개수 부분 조정해주면 됨


- Filter = [2, 2, 1, 3]
  - (w, h, channel, n)
  -  2 x 2 크기, 1 색깔, 3 개 필터

이게 Convolution의 힘!

필터 k개 써서 하나의 입력 이미지로부터 k개의 다른 형태의 이미지 뽑아낼 수 있어


(예) 
필터 3개 사용
```

┌ ─ ─ ─ ─┐    ┌ ─ ─ ─ ─┐    ┌ ─ ─ ─ ─┐
│ 1   1  │    │ 10  10 │    │ -1  -1 │
│ 1   1  │    │ 10  10 │    │ -1  -1 │
└ ─ ─ ─ ─┘    └ ─ ─ ─ ─┘    └ ─ ─ ─ ─┘
```

```
Rank = "[" 개수 = 4
Shape 셀 땐 맨 뒤부터 3 개 -> 1 개 ->  2개 -> 2 개 = (2, 2, 1, 3)
[
   [
      [[1.,10.,-1.]],
      [[1.,10.,-1.]]
   ],
   [
      [[1.,10.,-1.]],
      [[1.,10.,-1.]]
   ]
]
weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])


(cf) 이전 1개 일때
[
   [
      [[1.]],
      [[1.]]
   ],
   [
      [[1.]],
      [[1.]]
   ]
]
weight = tf.constant([[[[1.]],[[1.]]],
                     [[[1.]],[[1.]]]])
```

```
# Image
print("image.shape", image.shape)

# Filter
# 이전 1개일때
# weight = tf.constant([[[[1.]],[[1.]]],
#                      [[[1.]],[[1.]]]])

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print("weight.shape", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

# 여러 개 변수 쓰면 헷갈릴 수 있어, 
# Shape 헷갈리면 Print tensor 해봐
print(conv2d)  # Tensor("Conv2D_16:0", shape=(1, 3, 3, 3), dtype=float32)



# Output
print("conv2d_img.shape", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3) # 출력값 시각화
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')


                      
# 필터개수 늘리면 Convolution 값들도 많아짐
('image.shape', (1, 3, 3, 1))
('weight.shape', TensorShape([Dimension(2), Dimension(2), Dimension(1), Dimension(3)]))
('conv2d_img.shape', (1, 3, 3, 3))
[[ 12.  16.   9.]
 [ 24.  28.  15.]
 [ 15.  17.   9.]]
[[ 120.  160.   90.]
 [ 240.  280.  150.]
 [ 150.  170.   90.]]
[[-12. -16.  -9.]
 [-24. -28. -15.]
 [-15. -17.  -9.]]
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Max Pooling

Subsampling 개념  
Min, Max, Avg 중 Max pooling 많이 사용 (CNN과 잘 동작됨)  

TensorFlow에선 다음 세 개만 설정하면 됨
1) ksize (kernel size) Pooling 용 필터사이즈
2) stride
3) padding


(예)  
Zero-padding 시켜서 Pooling 하기

입력 이미지
```
┌ ─ ─ ─ ─┐
│ 4   3  │
│ 2   1  │
└ ─ ─ ─ ─┘
```

Shape 셀 땐 맨 뒤부터 1 개 -> 2 개 ->  2개 -> 1 개 = (1, 2, 2, 1)
```
[
   [
      [
         [4],
         [3]
      ],
      [
         [2],
         [1]
      ]
   ]
]
```

- tf.nn.max_pool
1) ksize   = [ 1, 2, 2, 1]  2 x 2
2) stride  = [.., 1, 1,..]  1 x 1
3) padding = SAME


Zero padding 한 이미지
```

┌ ─ ─ ─ ─ ─ ─┐
│ 4   3   0  │
│ 2   1   0  │
│ 0   0   0  │
└ ─ ─ ─ ─ ─ ─┘
```

Max pooling 과정
```
1) 
┌ ─ ─ ─ ─ ─ ─┐         ┌ ─ ─ ─ ─┐
│ 4   3   -  │         │ 4      │
│ 2   1   -  │    ->   │        │
│ -   -   -  │         └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

2) 
┌ ─ ─ ─ ─ ─ ─┐         ┌ ─ ─ ─ ─┐
│ -   3   0  │         │ 4   3  │
│ -   1   0  │    ->   │        │
│ -   -   -  │         └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

3) 
┌ ─ ─ ─ ─ ─ ─┐         ┌ ─ ─ ─ ─┐
│ -   -   -  │         │ 4   3  │
│ 2   1   -  │    ->   │ 2      │
│ 0   0   -  │         └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘

4) 
┌ ─ ─ ─ ─ ─ ─┐         ┌ ─ ─ ─ ─┐
│ -   -   -  │         │ 4   3  │
│ -   1   0  │    ->   │ 2   1  │
│ -   0   0  │         └ ─ ─ ─ ─┘
└ ─ ─ ─ ─ ─ ─┘
```


Max pooling 결과
```
┌ ─ ─ ─ ─┐
│ 4   3  │
│ 2   1  │
└ ─ ─ ─ ─┘
```

```
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)

# Max pooling
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME') # zero padding
print(pool.shape)
print(pool.eval())

# 여러 개 변수 쓰면 헷갈릴 수 있어, 
# Shape 헷갈리면 Print tensor 해봐
print(pool)  # Tensor("MaxPool_6:0", shape=(1, 2, 2, 1), dtype=float32)


(1, 2, 2, 1)
[[[[ 4.]
   [ 3.]]

  [[ 2.]
   [ 1.]]]]




# padding 없이 max pooling 하기
image = np.array([[
                   [[1],[2],[3]],
                   [[4],[5],[6]], 
                   [[7],[8],[9]]
                 ]], dtype=np.float32)

# Max pooling
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
print(pool.shape)
print(pool.eval())
rint(pool)

(1, 2, 2, 1)
[[[[ 5.]
   [ 6.]]

  [[ 8.]
   [ 9.]]]]
   
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### TensorFlow: max_pool

```
max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```

- value
  - 입력 데이터 (tf.float32 타입)
  - shape = [batch, in_height, in_width, in_channels]
  - batch : 데이터 갯수
  - in_height : row 수
  - in_width : col 수
  - in_channels : 채널 수
  - (예) 1, 3, 3, 1   (3 x 3 x 1) 의 1색상 이미지
	
	
- ksize
  - length >=4 인 int 리스트 형태로 선언
  - 풀링을 하는 sliding window의 크기를 의미
  - The size of the window for each dimension of the input tensor
  - (예) [1, 2, 2, 1] = [one image, width, height, one channel] 
  - pooling을 할때, 각 batch 에 대해 한 채널에 대해서 하니까, 1, 1,로 설정해준것
  - 2 x 2 크기이고 출력값은 1 개이다

- strides
  - length >=4 인 int 리스트 형태로 선언 [1, ?, ?, 1]
  - pooling 할때 sliding window의 이동을 얼만큼씩 할건지
  - The stride of the sliding window for each dimension of the input tensor.
  - (예) [1, 1, 1, 1]  1칸씩 이동

- padding
  - "VAILD", "SAME" 
  - VALID: 패딩 사용X, 데이터 사이즈 줄어듬
  - SAME : 패딩 사용O. 입력 데이터 사이즈가 유지

- data_format
  - 데이터의 포멧 ( default : "NHWC" )
  - NHWC [batch, height, width, channels]
  - NCHW [batch, channels, height, width]

- name
  - max_pool operation name 지정 (optional)

<br />
참조 링크
https://www.tensorflow.org/api_docs/python/tf/nn/max_pool


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### MNIST image loading

```
import os
os.chdir('/home/testu/work') 
os.getcwd()


from tensorflow.examples.tutorials.mnist import input_data #/home/testu/work/MNIST_data 에 다운 받음
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset


# 다운 받은 이미지 중 제일 첫번째 (images[0]) 것을 28 x 28 로 reshape 후 출력
img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')



# [CONV] Convolution layer
sess = tf.InteractiveSession()

img = img.reshape(-1,28,28,1) # 여러개 이미지 있으니 -1, 28 x 28 사이즈, 1가지 색상

W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01)) # 필터 크기 3 x 3, 1가지 색, 5개 사용

# stride (2 x 2)  필터 2 칸씩 옮기겠다
# zero padding 해서 처리할 것
# stride = 1 이면 입력 크기와 같아짐
# stride = 2 이니 입력 크기의 반으로 출력 -> (28 x 28) 입력이니 (14 x 14) 출력됨
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME') 
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

# 테스트용 이미지 출력
# 5개 필터를 사용해 출력된 convolution layer 의 결과 보기
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')



# [POOL] Max pooling
# ksize = sliding window 크기 (2 x 2)
# strides = (2 x 2)  2칸씩 움직일 것
# padding = SAME     zero padding 처리  (strides = 2 니 출력크기는 입력 이미지의 반으로 줄어)
# 입력 (14 x 14) -> 출력 (7 x 7)
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

# subsampling 된 후라 크기도 작아지고, 해상도 떨어져 보임
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

```

