<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

## 01. Introduction

<br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Hello world

Docker TensorFlow 시작

```
docker start tftest
docker exec -it tftest bash
```

Jupyter 실행
```
import tensorflow as tf
tf.__version__

# Node = mathematical operation
# Edge = tensor (data array)


# Generate a node (constant operation) named "hello"
hello = tf.constant("Hello, TensorFlow.")

# Start a tf session
sess = tf.Session()

# run the op: sess.run()
# get result: print
print(sess.run(hello))

#b'Hello, TensorFlow.'
#b: Bytes literals
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### TF Mechanics (3 steps)

TF 알고리즘은 크게 3단계로 구현
```
import tensorflow as tf

# 1) Build a graph using tf operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)     # tf.float32 implicitly
node3 = tf.add(node1, node2) # or node3 = node1 + node2

print("node1", node1, "node2", node2)
print("node3", node3)
# ('node1', <tf.Tensor 'Const_1:0' shape=() dtype=float32>, 'node2', <tf.Tensor 'Const_2:0' shape=() dtype=float32>)
# ('node3', <tf.Tensor 'Add:0' shape=() dtype=float32>)

# 2) Feed data and run graph (operation) using sess.run(op)
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))
# ('sess.run(node1, node2): ', [3.0, 4.0])
# ('sess.run(node3): ', 7.0)

# 3) Update variables in the graph (and return values)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Placeholder

constant 대신 그래프 실행 단계에서 값 던질수 있음  
placeholder 인 node 사용

```
import tensorflow as tf

# 1) Build a graph using tf operations
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # or adder_node = tf.add(a, b)

# 2) Feed data and run graph (operation) using sess.run(op, feed_dict={x: x_data})
# feed_dict 로 node a, b에 값 넘겨줌
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2,4]}))
# 7.5
# [ 3.  7.]

# 3) Update variables in the graph (and return values)
```


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Operations

assign(), add() [[1]] 등의 tf operations [[2]] 들은 expression graph의 일종 [[3]]  
sess.run() 으로 실행하기전에는 실제 연산 수행하지 않음

```
# Define constants
a = tf.constant([3, 6])
b = tf.constant([2, 2])

sess = tf.Session()

# Addition
print(sess.run(tf.add(a, b))) # >> [5 8]
print(sess.run(tf.add_n([a, b, b]))) # >> [7 10]. Equivalent to a + b + b 
      
# Multiplication
print(sess.run(tf.multiply(a, b))) # >> [6 12] because mul is element wise 
      
# Matrix multiplication
# tf.matmul(aa, bb) # >> ValueError
print(sess.run(tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1])))) # >> [[18]] 

# Division
print(sess.run(tf.div(a, b))) # >> [1 3] tf.mod(a, b) # >> [1 0]
```

[1]: https://www.tensorflow.org/api_docs/python/tf/add "https://www.tensorflow.org/api_docs/python/tf/add"
[2]: https://www.linkedin.com/pulse/tensorflow-operations-kaustubh-narkhede "https://www.linkedin.com/pulse/tensorflow-operations-kaustubh-narkhede"
[3]: https://codeonweb.com/entry/5f15bf8e-d704-49e0-909a-db4450433b74 "https://codeonweb.com/entry/5f15bf8e-d704-49e0-909a-db4450433b74"



<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Everything is Tensor

Tensor는 일종의 array [[4]]  
<br />

1) Rank: 몇 차원 array 인가

| Rank | Math entity                       | Python example                                        |
| :--: | :-------------------------------- | :---------------------------------------------------- |
| 0    | Scalar   (magnitude only)         | s = 483                                               |
| 1    | Vector   (magnitude and direction)| v = [1.1, 2.2, 3.3]                                   |
| 2    | Matrix   (table of numbers)       | m = [[1,2,3], [4,5,6], [7,8,9]]                       |
| 3    | 3-Tensor (cube of numbers)        | t = [[[2],[4],[6]], [[8],[10],[12]], [[14],[16],[18]]]|
| n    | n-Tensor (you get the idea)       |                                                       |
<br />

2) Shape: 몇 개 element 있나 (tf 개발시 shape 설계 중요)

| Rank | Shape             | Dim. | Example                                                     |
| :--: | :---------------- | :--: | :---------------------------------------------------------- |
| 0    | [ ]               | 0-D  | 0-D tensor = scalar                                         |
| 1    | [D0]              | 1-D  | 1-D tensor with shape [2]   : [1., 2.]                      |
| 2    | [D0, D1]          | 2-D  | 2-D tensor with shape [2, 3]: [[1., 2., 3.], [4., 5., 6.]]  |
| 3    | [D0, D1, D2]      | 3-D  | 3-D tensor with shape [2,1,3]: [[[1.,2.,3.]], [[7.,8.,9.]]] |
| n    | [D0, D1,.., Dn-1] | n-D  | a tensor with shape   [D0, D1,.., Dn-1]                     |
<br />

Example
```
3                                # rank 0 tensor: shape [ ]      scalar 
[1., 2., 3.]                     # rank 1 tensor: shape [3]      vector 
[[1., 2., 3.], [4., 5., 6.]]     # rank 2 tensor: shape [2, 3]   matrix 
[[[1., 2., 3.]], [[7., 8., 9.]]] # rank 3 tensor: shape [2, 1, 3] 
```
<br />

3) Types: 데이터타입

대부분 tf.float32 사용 [[5]]  
정수는 tf.int32 주로 사용 

```
tf.float32
tf.float64
tf.int8
tf.int16
tf.int32
tf.int64
```


[4]: https://www.tensorflow.org/programmers_guide/dims_types "https://www.tensorflow.org/programmers_guide/dims_types"
[5]: https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow "https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow"
