<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 07. Application & Tips


I. �ӽŷ��� �˰��� ���� �� ����

1) Learning rate �����ϴ� ��
2) Data preprocessing �ϴ� ��
3) Overfitting �����ϴ� ��!!


II. �ӽŷ��� ��� �󸶳� �� �����ϴ��� Ȯ���ϴ� ���  
Performance evaluation


���� ��ũ  
https://www.youtube.com/watch?v=1jPjVoDV_uo&feature=youtu.be


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-1) Learning rate �����ϴ� ��

gradient descent �� learning_rate ���Ƿ� ���� �ǽ��߾�
```
learning_rate = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypo), axis = 1))  # axid ��� reduction_indices �ᵵ �Ǵ���
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)
```

learning_rate �� ���ϴ°� �߿�
- Large learning rate
  - overshoot �ʹ� ���沱�� �پ�, - + �Դٰ���, cost �پ���� �ʰ� ȮĿ���� ������ Ƣ���, �� �ٿ���
- Small learning rate
  - takes too long, stops at local minimum, �����ߴ��� �۰� ���ϴ����ϸ� �� ���� �÷���


Try several learning rates Ư���� ���� ���� (������, ���� ȯ�� ���� �� �޶�)  
Observe the cost function  
Check it goes down in a reasonable rate  
���� 0.01 �� ����, cost �Լ� ���鼭 ����  


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-2) Data preprocessing �ϴ� ��

Data (X) preprocessing for gradient

x1 �� 1~10 ���̰�, x2 �� 1000 ~ 10000 ���� ���̸�  
w1, w2 �� �������� Ŀ  
learning rate�� w2 �н��ϴµ� ���� ���̾ w1���� �ʹ� Ŀ�� ���沱�� �ٰ� ���� �� �־�  
normalize �ʿ�!


original data -> zero-centered data -> normalized data

original data �� skewed �������� zero-centered data ����  
���� �ʹ� ����� (Ȥ�� ������) ������ �ϴ� ���� �ذ� ����

normalized data�� x1 x2 ���� ���� ���ָ� ����

�ϴ� ���  

- Standardization
  - x' = (x - ��) / �� 
  - �� [��] ���
  -�� [�ñ׸�] ǥ������

```
x_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 1-3) Overfitting �����ϴ� ��

�ӽŷ����� ���� ū ����!  
Ʈ���̴� �����ʹ� �� ���߳� ���� ������ ������ �ȸ¾�  


Ʈ���̴� �����͸� ���� ���ϱ� (�������� overfitting ���� �� )  
feature ���� ���̱� (�ߺ��Ȱ� ���̴� ��)  
Regularization (�Ϲ�ȭ)  


#### Regularization  
weight �ʹ� ū �� ������ ����

2���� �� data ���� �� �̸� classification�ϴ� decision boundary��   
�ʹ� �� ���߷��� wavy�ϰ� �Ѵٴ� ���� weight�� ũ�ٴ� �ǹ�  
weight �۰��ϸ� decision boundary ���� ���ұ����Ѱ� ���� ����  

cost function �� �ϳ� �� �߰�

L = 1/N ��(D(S(WX + b), L) + ���W^2

- ���W^2
  - �� w element�� �����ؼ� �� ������
  - ��: ���, regularization strength  (0�̸� regularization ���ϰڴ�, Ŭ���� �� �� ����Ŀ��)  

cost �Լ� �̺н��� gradient descent ���Ҷ� - ���L  
���W^2 �� �̺� �Ǹ鼭 ��L�� �ٿ���

 
�ڵ� ��
```
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
cost = tf.reduce_mean(tf.square(h - Y)) + l2reg
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

(cf) ������ ������ �ӽŷ��� ����  
chap 4. artificial neural network

������ �κ� Alternative error functions
1) Penalize large weights (weight decay) �����


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Performance evaluation

training data�� -> ML model �н�����

�׽�Ʈ�� training data �״�� ���� NG (�翬�� �� �´ٰ� �� ��)


data: 70% training set, 30% test set ���� ��� [[1]]

[1]: http://www.holehouse.org/mlclass/10_Advice_for_applying_machine_learning.html

�� Ȥ�� ��[��Ÿ]: learning rate  
��: regularization strength  

�̷��� Ʃ������

training set �� �ٽ� training, validation ������ ����  

training set���� �н�  
validation ���� ���ǽ����ؼ� ��,��,�� �� üũ


(��)  
MINIST Dataset  

���� �ʱ�ü �νĿ� ������ (��ü������ �����ȣ �ĺ� ����)
(http://yann.lecun.com/exdb/mnist)

```
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```
�� Ʈ���̴�, �׽�Ʈ �����ͷ� ������ �־�  


(cf)  
Online learning ���

������ ������ �ѹ��� �� ����ϱ� �����

- training set 100���� -> Model 
  - �߶� �н�
  - 10���� -> Model �н�
  - 10���� -> Model �߰� �н� (���� 10�� �� �н� ����� �н��� ����)
  - 10���� -> Model �߰� �н�
  - ...


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Accuracy

���� ������ Y ���� ���� �����Ѱ� ��

10���� 7�� ���߸� 70%


