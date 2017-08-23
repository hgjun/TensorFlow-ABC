<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 12-1. RNN Introduction

Recurrent Neural Network


<br />  
참조 링크  
https://www.youtube.com/watch?v=-SHPG_KMUkQ&feature=youtu.be  

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Sequence Data

한 문장은 단어 하나만 이해하면 X, 이전의 단어들도 알아야

NN, CNN은 sequence data 처리 힘들어

```
       Hi
        ↑
        │      
.. ─→ [   ]─→ .. 현재 state가 다음 state에 영향 미친다
        ↑
        │
        Xi
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Recurrent Neural Network

state 개념 있어

x: sequence of vectors  
h: state  
f: function with parameters W

(h_t 표시는 h 아래 첨자에 t 있음 의미)

h_t = f_w(h_t-1, x_t)

```
          y
          ↑               ↑
 h_t-1    │     h_t       │
 ────→ [ RNN ] ─────→  [ RNN ] ────→ .
          ↑               ↑
          │               │
          x_t
```

그런데 보통 RNN은 다음처럼 그림 하나로 표현해
```
           y
           ↑
           │  ┌───┐
        [ RNN ] <─┘
           ↑ 
           │
           x_t
```
=> 함수 f_w 가 모든 RNN에서 동일하게 사용하기 때문!


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### (Vanilla) Recurrent Neural Network

가장 기초적인 RNN 계산 방법  
state는 hidden vector h

WX 형태로 f 구성함

```
h_t = f_w(h_t-1, x_t)  

    = tanh(W_hh * h_t-1  + W_xh * x_t)
```
    
W_hh 는 h_t-1 의 weight  
W_xh 는 x_t   의 weight  
tanh는 sigmoid 함수  
계산 결과는 다음 state에 넘겨줘

y_t = W_hy * h_t

계산된 h_t 에 weight값 W_hy 곱해서 계산
 

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Character-level Language Model Example

Vocabulary: [h, e, l, o]

Example training sequence: "hello"

(예)
사용자가 검색창에 hell 까지 쳤으면,   
자동으로 완성된 단어를 만들어서  
추천해주기 위해 그 다음 어떤 문자를 붙여줘야 할까

```
   h0      h1      h2      h3      
    ↑       ↑       ↑       ↑
  [ A ]   [ A ]   [ A ]   [ A ]
    ↑       ↑       ↑       ↑
   x0      x1      x2      x3
   "h"    "e"      "l"     "l"
```

#### 0) 입력 문자 one-hot encoding

```
input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l
```

이 값을 RNN set에 입력

#### 1) h_1 계산

h_t = tanh(W_hh * h_t-1  + W_xh * x_t)

보통 가장 처음 h_t-1 는 0 으로 둬

h_t = tanh(W_hh * 0  + W_xh * x_t)

```

hidden layer   0.3  
              -0.1  
               0.9  

              ↑ W_xh

input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l

```

#### 2) h_2 계산

h_t = tanh(W_hh * h_t-1  + W_xh * x_t)

h_1 과 x2 둘 다 사용해 계산

[0.3, -0.1, 0.9] * W_hh  + [0 1 0 0] * W_xh => [1.0, 0.3, 0.1]
```

hidden layer   0.3  W_hh   1.0
              -0.1   →    0.3
               0.9         0.1

              ↑           ↑ W_xh

input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l

```

#### 3) h_3 계산

h_t = tanh(W_hh * h_t-1  + W_xh * x_t)
```

hidden layer   0.3         1.0   W_hh   0.1
              -0.1   →    0.3    →   -0.5
               0.9         0.1         -0.3

              ↑           ↑           ↑ W_xh

input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l

```

#### 4) h_4 계산

h_t = tanh(W_hh * h_t-1  + W_xh * x_t)
```
                                     이전의 값들이 지금 값에 영향주고 있어
                                     
hidden layer   0.3         1.0          0.1   W_hh  -0.3
              -0.1   →    0.3    →   -0.5   →     0.9
               0.9         0.1         -0.3          0.7

              ↑           ↑           ↑            ↑ W_xh

input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l

```

#### 5) 마지막으로 y값 계산
y_t = W_hy * h_t
```
                         0010 원했는데 0001 나왔어
                         이 cost 계산은 softmax 등 이전 cost func 써서 학습하면되
                         얘는 틀렸어
target chars   "e"         "l"          "l"           "o"   원하는 결과

output layer   1.0         0.5          0.1           0.2
               2.2 ok      0.3          0.5          -1.5
              -3.0        -1.0 원해     1.9  ok      -0.1
               4.1         1.2 NG      -1.1           2.2  ok
               
                ↑ W_hy     ↑ W_hy     ↑ W_hy       ↑ W_hy
             state 값 (h_t) 들에 일괄적으로 W_hy 곱해서 계산        

hidden layer   0.3         1.0          0.1   W_hh  -0.3
              -0.1   →    0.3    →   -0.5   →     0.9
               0.9         0.1         -0.3          0.7

              ↑           ↑           ↑            ↑ W_xh

input layer    1            0            0            0
               0            1            0            0
               0            0            1            1
               0            0            0            0
               
input chars    h            e            l            l

```

<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### RNN Applications

활용분야 굉장히 많아  
https://github.com/TensorFlowKR/awesome_tensorflow_implementations


- Language Modeling    연관검색어
- Speech Recognition   스피치 인식
- Machine Translation  기계번역 (시퀀스 데이터 형식이니)
- Conversation Modeling/Question Answering   챗봇 개발
- Image/Video Captioning   자막
- Image/Music/Dance Generation


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Recurrent Network

여러 형태로 구성가능

| Types        | Examples                                                   |
|--------------|------------------------------------------------------------|
| one to one   | Vanilla Nerual Networks                                    |
| one to many  | image captioning: 이미지 하나 -> sequence of words         |
| many to one  | Sentiment Classification: sequence of words -> 기쁨 (감정) |
| many to many | Machine Translation: seq of words -> seq of words          |
| many to many | Video classification on frame level: 각각의 프레임 -> 설명 |


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Multi-layer RNN

아까 hidden layer 하나였어, 
```
    [ ]    [ ]    [ ]    [ ]
```

layer 늘려


```
depth 늘려
↑
    [ ]    [ ]    [ ]    [ ]
    
    [ ]    [ ]    [ ]    [ ]

    [ ]    [ ]    [ ]    [ ]
      → time
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Training RNN is Challenging

RNN도 deep 해져서 layer 늘어나면 학습 어려워져

RNN 과 같은데 조금 다른 모델들 나오고 있어
 
- Long Short Term Memory (LSTM)  
- GRU [Cho er al. 2014]

요샌 RNN 안쓰고 이런것들 사용하고 있음






