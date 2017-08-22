<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 11-1. CNN Introduction


기본 네트워크는 Forward (Fully connected)
```
x - [ ] - [ ] - [ ] - [ ] - [ ] - [ ] -  y-hat
```

다음 형태가 Convolutional NN의 기본 아이디어
```
x1 [ ] - [ ] - [ ]  ┐
x2 [ ] - [ ] - [ ]  ┼-- [ ] - [ ] -  y-hat
x3 [ ] - [ ] - [ ]  ┘          
```

<br />
참조 링크  
https://www.youtube.com/watch?v=Em63mknbtWo&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Convolution

#### 이미지

<div class="image_box"></div>
32 x 32 x 3 image (3은 RGB 칼라정보)


-> 이전에는 32 x 32 x 3 다 입력으로 주고 32 x 32 x 3 개의 weight 학습시켰어  
-> small area 로 나누어 처리!! 이것이 convolution nn의 아이디어


#### 필터

<div class="image_box_small"></div>
5 x 5 x 3 filter (필터크기는 사용자 정의 가능, 색 처리는 3으로 같아)

이 필터로 하나의 값을 만들어내 (one number)

5 x 5 받아서 one number 로 만들어내는 과정

= Wx + b

w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + b = y_hat 하나의 숫자 만들어!

w1 ~ w5 가 어떤 숫자 하나 (y_hat)를 결정하는 값이야

#### Convolution layer

필터 거친 결과로 Conv layer 나옴


ReLU 쓰고싶으면 다음처럼 하면돼

= ReLU(Wx + b)

계속 이미지 내에서 필터 움직이며 하나의 값들을 만들어가


(cf) 컴퓨터 비전쪽에서 convolution은 filtering이란 의미임


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Filter

#### Stride
필터가 움직이는 칸 크기
```
이미지크기 N: 7 X 7  
필터크기   F: 3 x 3   
출력크기   O: 5 x 5   
```

출력크기 = (N - F) / stride + 1
```
stride 1 = > (7 - 3)/1 + 1 = 5
stride 2 = > (7 - 3)/2 + 1 = 3
stride 3 = > (7 - 3)/3 + 1 = 2.33  (그래서 stride 3은 사용할 수 없어)
```

즉 건너뛸수록 정보 잃어버리는 양 커짐


#### Padding

모서리, border line에 있는 픽셀정보도 안에 있는 정보와 공평하게 필터링하기 위함

(예)
```
input   7 x 7
filter  3 x 3
stride  1
pad with 1 pixel border
```
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|---|---|---|---|---|---|---|---|---|
| 0 |   |   |   |   |   |   |   | 0 |
| 0 |   |   |   |   |   |   |   | 0 |
| 0 |   |   |   |   |   |   |   | 0 |
| 0 |   |   |   |   |   |   |   | 0 |
| 0 |   |   |   |   |   |   |   | 0 |
| 0 |   |   |   |   |   |   |   | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


패딩 후에 9 x 9 되 =>  (9 - 3)/1 + 1 = 7  원래 이미지와 같은 크기 됨! 7 x 7

이게 하나의 convolutional layer 만드는 방법


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Convolution Layer

입력 이미지

<div class="image_box"></div>

32 x 32 x 3 image 

필터링

<div class="image_box_small"></div>
5 x 5 x 3 filter 6개 (각각 weight 달라서 다른 6개의 output 나올 것)


Convolution layer
<div class="image_box_mid"></div>

(32 - 5)/1 + 1 = 28  

계산후 activation maps  

패딩안하면 (28, 28, 6) 나와



#### 여러 Convolution layers 사용 가능

(32 x 32 x 3) image

-> Conv layer1: Conv, ReLU 같이 사용 (6개 5 x 5 x 3 filters)  

-> (28, 28, 6) activation maps  

-> Conv layer2: 여기에 다시 Conv, ReLU (10개 5 x 5 x 3 filters)

-> (24, 24, 10) activation maps ((28 - 5)/1 + 1 = 24)

-> ...
     
   
   
그럼 weight 개수는? 
```
(5 x 5 x 3) x 6 개 +
(5 x 5 x 3) x 10 개 + ...
```
CNN 이전 방법은 image 픽셀 수 만큼 필요했었는데, 그 수 줄어들었어!

처음엔 randomly initial value 주고 학습시킴


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 11-2. CNN Introduction

Max Pooling and Full Network


<br />
참조 링크  
https://www.youtube.com/watch?v=2-75C-yZaoA&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Max Pooling

input images  ->  convolution layer (사용한 filter 개수 따라 depth 달라졌었어)

-> Pooling layer (sampling 으로 보면 됨)  
앞의 convolution layer에서 한 layer씩 뽑아서 resize 하는 것이 pooling (depth 만큼 반복)

(예)  
4 x 4 이미지 -> 2 x 2 필터, stride 2 면 (총 4번 움직임)  
(4 - 2)/2 + 1 = 2 이므로 2 x 2 output 만들어짐

필터 방법은 여러가지
```
1 1 2 4
5 6 7 8
3 2 1 0
1 2 3 4
```

처음
```
1 1
5 6
```

max 필터면 6  
min 필터면 1  
avg 필터면 6.5 계산됨  
그중 가장 많이 쓰이는게 max pooling

Max pooling 결과는
```
6 8
3 4
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### CNN Full Network

#### Convolution 과정

이미지 -> Conv Layer -> ReLU(Conv) -> Conv -> ReLU -> Pool

방법은 여러가지

이미지 -> Conv -> ReLU -> Pool -> ...



#### Convolution 마지막엔 보통 Pooling 

이미지 -> Conv -> .... -> Pooling

(예)  
pooling output 3 x 3 x 10  ==> 이게 이제 다시 input X 가 되!



#### FC Layer (Fully Connected Layer)

Contains neurons that connect to the entire input volume, as in ordinary Neural Networks  
unit끼리 모두 연결된 구조의 NN  
FC Layer 로 Convolution layer 학습시킨 거를 마무리하는데 사용

이전 결과를 입력 X로 받아서 일반적인 NN 혹은 FC Layer 2개 정도 돌려

이미지 -> Conv -> .... -> Pool -> [FC Layer] -> [FC Layer]


#### 맨 마지막 Softmax Classifier 
```
이미지 -> Conv -> .... -> Pool -> [FC Layer] -> [FC Layer] -> [Softmax] -> Y_hat
```
#### 결과
one-hot encoded 형태 
[차, 트럭, 비행기, 배, 말]
[1, 0, 0, 0, 0] 이면 "차"로 계산한 것


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### ConvNetJS demo: Training on CIFAR-10


http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html



<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 11-3. CNN case study


지금까진 CNN 기본  
이제 응용버전, 이것들 어떻게 사용할지 볼 것


참조 링크  
https://www.youtube.com/watch?v=KbNbWTnlYXs&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### LeNet-5
LeCun et al., 1998

Input img 32 x 32 x 1 (검은색) 

-> Convolution (28 x 28 x 6)     
conv filter: 5 x 5, stride 1, 6개  
(32 - 5) / 1 + 1 = 28

-> Subsampling (14 x 14 x 6)  
pooling    : 2 x 2 stride 2  
(28 - 2) / 2 + 1 = 14

-> Convolution (10 x 10 x 16)     
conv filter: 5 x 5, stride 1, 16개  
(14 - 5) / 1 + 1 = 10

-> Subsampling (5 x 5 x 16)
pooling    : 2 x 2 stride 2  
(10 - 2) / 2 + 1 = 5

-> FC layer (120)

-> FC layer (84)

-> Gaussian connections (10 output)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### AlexNet

Krizhevsky et al. 2012  
imagenet 이라는 경진대회에서 1등

ReLU 처음 사용  
Normalization layer 사용 (지금은 안씀, 굳이 안해도 되더라)  
heavy data augmentation  
dropout 0.5  
batch size 128  
SGD Momentum 0.9  
Learning rate 1e-2, reduced by 10 manually when val accuracy plateaus  
L2 weight decay 5e-4  

7 CNN ensemble: 위 구조 7개 만들어서 그 결과 합쳐 (한개만 사용시 오류율 18.2% -> 합친결과 15.4%)


input (227 x 227 x 3)

-> Conv1: 1st layer (output volume: 55 x 55 x 96)  
96개 11 x 11 filters, stride 4, pad 0  
(227 - 11) / 4 + 1 = 55

35K의 변수(파라메터 필요)  
Parameters: (11 * 11 * 3) * 96 = 35K  


-> Pool1: 2nd layer (27 x 27 x 96)  
3 x 3 filters, stride 2  
Parameters: 0  (Pooling이니 특별히 변수 필요 없어)

-> Norm1: 3rd layer (27 x 27 x 96)  
앞에서 받은 값을 normalizer 하는 레이어  
최근엔 사용안함 (굳이 안해도 되더라)
..



#### Full (simlified) AlexNet architecture:
```
[227x227x  3] INPUT

[ 55x 55x 96] CONV1    : 96 11 x11 filters, stride 4, pad 0
[ 27x 27x 96] MAX POOL1:     3 x 3 filters, stride 2
[ 27x 27x 96] NORM1    : Normalization layer

[ 27x 27x256] CONV2    : 256 5 x 5 filters, stride 1, pad 2
[ 13x 13x256] MAX POOL2:     3 x 3 filters, stride 2
[ 13x 13x256] NORM2    : Normalization layer

[ 13x 13x384] CONV3    : 384 3 x 3 filters, stride 1, pad 1
[ 13x 13x384] CONV4    : 384 3 x 3 filters, stride 1, pad 1
[ 13x 13x256] CONV5    : 256 3 x 3 filters, stride 1, pad 1
[  6x  6x256] MAX POOL3:     3 x 3 filters at stride 2

[4096] FC6: 4096 neurons  (입력 6x6x256 = 9216 , 출력 4096)
[4096] FC7: 4096 neurons  (입력 4096 , 출력 4096)
[1000] FC8: 1000 neurons  (입력 4096 , 출력 1000) class scores, 1000개 이미지로 물체 분류
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### GoogLeNet

Szegedy et al., 2014  
ImageNet (ILSVRC) 2014년 우승

Inception module with dimension reductions  
레이어 늘려 깊게 층 만들면 banishing gradients 문제 생겨   
이걸 1 x 1 convolution 사용해 차원을 줄여 해결

AlexNet 보다 깊이는 깊은데 free parameter 는 1/12 수준, 연산량도 적어  

(http://laonple.blog.me/220686328027)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### ResNet

He et al., 2015  
홍콩중문대 박사, MS Research 근무  
ImageNet (ILSVRC) 2015년 우승 (3.6%, 사람의 오류율 5%보다 작아졌어)  
그 외 다른 대회에서도 우승

(https://www.youtube.com/watch?v=1PGLj-uKT1w)



AlexNet (ILSVRC 2012 우승) 8 layer

VGG     (ILSVRC 2014 우승) 19 layer

ResNet  (ILSVRC 2015 우승) 152 layer 사용!

8 GPU 머신에서 2 ~ 3 주 학습시킴  
VGG보다 8배 많은 레이어임에도 불구하고  
Run time 에서는 VGGNet 보다 빨랐어

-> Fast Forward 개념 사용  
TensorFlow ex - lab10. nn2 2) dropout, ensemble, nn construction.txt  135 line 참조


Plaint Net
```
      x       
      ↓      
[weight layer]

      ↓ relu
      
[weight layer]

      ↓ relu

     H(x)
```
구조에서 다음으로 변경했어



Residual Net
```
      x

      ↓============┐
     F(x)          │
[weight layer]     │
                   │
      ↓ relu       │ identity x  점프에서 더해져
                   │  
[weight layer]     │
      ↓            │
       <<==========┘ 여기까지가 하나의 레이어처럼 돼
      ↓ relu 

 H(x) = F(x) + x
```


전체 레이어는 크지만,  
실제 학습하는 입장에서는 레이어가 깊지 않은 느낌으로 학습할 수 있는 것이  
잘 되는 이유중 하나일 것

아직 잘 되는 이유는 정확히 이해 안되고 있어


ResNet 도 GoogLeNet 처럼 1 x 1 conv 로 dim reduction 해줌
```
     x
     ↓=========┐
[1 x 1,  64]   │
     ↓ relu    │ 
[3 x 3,  64]   │
     ↓         │
[1 x 1, 256]   │
       <<======┘
     ↓ relu 
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Sentence Classification

CNN은 이미지 이 외에도 처리 잘한다

Convolutional Neural Networks for Sentence Classification  
[Yoon Kim, 2014]

NLP, Sentence 처리에 CNN 사용


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### AlphaGo

CNN 사용

policy network:

Input: [19x19x48] 48개의 feature plane (돌이 있다, 없다, ...)

CONV1     : 192 5x5 filters, stride 1, pad 2 => [19x19x192]

CONV2 ~ 12: 192 3x3 filters, stride 1, pad 1 => [19x19x192]

CONV      :   1 1x1 filter , stride 1, pad 0 => [19x19] (probability map of promising moves)




<style>
div.image_box {
    width: 100px;
    height: 100px;
    border: 1px solid #000000;
    padding: 5px;
    margin: 10px;
}
div.image_box_mid {
    width: 80px;
    height: 80px;
    border: 1px solid #000000;
    padding: 5px;
    margin: 10px;
}
div.image_box_small {
    width: 40px;
    height: 40px;
    border: 1px solid #000000;
    padding: 5px;
    margin: 10px;
}
</style>
