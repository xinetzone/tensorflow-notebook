## TensorFlow 基础

如果您想要在 Tensorflow 中运行 TensorFlow1.X，可以这样：

```python
import tensorflow.compat.v1 as tf

tf.disable_eager_execution() # 关闭 Eager Execution 模式
```

TensorFlow 默认模式是即时执行模式（Eager Execution）。载入模块：

```python
import tensorflow as tf
```

### 数据流图

TensorFlow 的计算表示为数据流图。`tf.function` 使用 graph 来表示函数的计算。每个 graph 包含一组 `tf.Operation` 对象（代表计算单元）和 `tf.Tensor` 对象（表示操作流之间流动的数据单元）。

可以使用 `tf.Graph.as_default` 上下文管理器注册默认 graph。 然后，将操作添加到 graph 中，而不是被立即执行。 例如：

```python
g = tf.Graph()
with g.as_default():
    # Define operations and tensors in `g`.
    c = tf.constant(30.0)
    assert c.graph is g
```

~~一个 `tf.Graph` 实例支持由 `name` 标识的任意数量的“集合”。为方便起见，在构建大型 graph 时，集合可以存储相关对象的组：例如，`tf.Variable` 使用一个集合（名为`tf.GraphKeys.GLOBAL_VARIABLES`）收集构建 graph 期间创建的所有变量。调用者可以通过指定新的 `name` 来定义其他集合。~~

`tf.Operation` 表示在张量上执行计算的 graph 节点。换言之，`tf.Operation` 是 `tf.Graph` 中的一个节点，该节点将零个或多个 `tf.Tensor` 对象作为输入，并产生零个或多个 `tf.Tensor` 对象作为输出。

可以通过在 `tf.function` 或 `tf.Graph.as_default` 上下文管理器中调用 Python op 构造函数（例如 `tf.matmul`）来创建类型为 `tf.Operation` 的对象。例如，在 `tf.function` 中，`c = tf.matmul（a，b）` 创建一个以张量 `a` 和 `b` 作为输入，并且产生 `c` 作为输出的类型为“MatMul” 的 `tf.Operation`。

`tf.Tensor` 表示 `tf.Operation` 的输出。

### 自动微分机制

在即时执行模式下，TensorFlow 引入了 `tf.GradientTape()` 这个 “求导记录器” 来实现自动求导。以下代码展示了如何使用 `tf.GradientTape()` 计算函数 $y(x) = x^5$ 在 $x = 7$ 时的导数：

```python
x = tf.Variable(initial_value=7.)
# 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
with tf.GradientTape() as tape:     
    y = x ** 5
y_grad = tape.gradient(y, x) # 计算 y 关于 x 的导数
tf.print(y, y_grad)
```

显示：

```js
16807 12005
```

也可以对多元函数求导：

```python
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
L, w_grad, b_grad
```

输出：

```js
(<tf.Tensor: shape=(), dtype=float32, numpy=125.0>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
 array([[ 70.],
        [100.]], dtype=float32)>,
 <tf.Tensor: shape=(), dtype=float32, numpy=30.0>)
```