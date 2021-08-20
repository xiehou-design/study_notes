# 数据封装

使用data.Dataset.from_tensor_slices函数来封装x和y

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))



x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))

做数据打乱处理，来增强数据的可用性

***dataset = dataset.shuffle(buffer_size=1024).batch(32)***
model.fit(dataset, epochs=1)



# 模型画结构图

***函数式API是构建结构图对应模型的方法***

<u>*`inputs = tf.keras.Input(shape=(784,), name='img')`*</u>

`以上一层的输出作为下一层的输入`

`h1 = layers.Dense(32, activation='relu')(inputs)`
`h2 = layers.Dense(32, activation='relu')(h1)`
`outputs = layers.Dense(10, activation='softmax')(h2)`
*`<u>model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')  # 名字字符串中不能有空格</u>`*

`model.summary()`
**`keras.utils.plot_model(model, 'mnist_model.png')`**
**`keras.utils.plot_model(model, 'model_info.png', show_shapes=True)`**



# 上采样

两种好像都是上采样的方法，但是可能存在一些区别，都是从小模版扩展到大模板，目标是使得输入和输出的形状基本相同，也就是如Unet模型一般，输入和输出大小一致。

tf.keras.layers.Conv2DTranspose

keras.layers.UpSampling2D

**请注意，我们使解码架构与编码架构严格对称，因此我们得到的输出形状与输入形状相同(28, 28, 1)。**
**Conv2D一层的反面是Conv2DTranspose一层，MaxPooling2D一层的反面是UpSampling2D一层。**



# 模型组合和嵌套

**模型可以嵌套**：模型可以包含子模型（因为模型就像一层一样）。

**模型嵌套的另一种常见模式是集成**。以下是将一组模型整合为一个平均其预测值的模型的方法：

`def get_model():`
    `inputs = keras.Input(shape=(128,))`
    `outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)`
    `return keras.Model(inputs, outputs)`

`model1 = get_model()`
`model2 = get_model()`
`model3 = get_model()`
`inputs = keras.Input(shape=(128,))`
`y1 = model1(inputs)`
`y2 = model2(inputs)`
`y3 = model3(inputs)`
***`outputs = layers.average([y1, y2, y3])`***
***`ensemble_model = keras.Model(inputs, outputs)`***



# ResNet模型

`inputs = keras.Input(shape=(32,32,3), name='img')`
`h1 = layers.Conv2D(32, 3, activation='relu')(inputs)`
`h1 = layers.Conv2D(64, 3, activation='relu')(h1)`
**`block1_out = layers.MaxPooling2D(3)(h1)`**

`h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(block1_out)`
`h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(h2)`
**`block2_out = layers.add([h2, block1_out])  # 残差连接`**

`h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(block2_out)`
`h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(h3)`
**`block3_out = layers.add([h3, block2_out])`**

`h4 = layers.Conv2D(64, 3, activation='relu')(block3_out)`
`h4 = layers.GlobalMaxPool2D()(h4)`
`h4 = layers.Dense(256, activation='relu')(h4)`
`h4 = layers.Dropout(0.5)(h4)`
`outputs = layers.Dense(10, activation='softmax')(h4)`

`model = keras.Model(inputs, outputs, name='small_resnet')  # 网络名不能有空格`
`model.summary()`
**`keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)`**

![image-20210806213518991](E:\Typora\project\small_ResNet)

# 自定义网络层
**tf.keras具有广泛的内置层**。这里有一些例子：

- 卷积层：Conv1D，Conv2D，Conv3D，Conv2DTranspose，等。
- 池层：MaxPooling1D，MaxPooling2D，MaxPooling3D，AveragePooling1D，等。
- RNN层：GRU，LSTM，ConvLSTM2D，等。
- BatchNormalization，Dropout，Embedding，等。
如果找不到所需的内容，则可以通过创建自己的图层来扩展API。

**所有层都对该Layer类进行子类化并实现：**

- 一个call方法，指定由该层完成的计算。
- 一种build创建图层权重的方法（请注意，这只是一种样式约定；也可以在__init__函数中创建权重）。



# 自定义损失损失函数

用Keras提供两种方式来提供自定义损失。

- 一、例创建一个接受输入y_true和的函数y_pred。

- 二、构建一个继承keras.losser.Loss的子类

  下面示例显示了一个损失函数，该函数计算实际数据与预测之间的平均距离：

  `def get_uncompiled_model():`
    `inputs = keras.Input(shape=(784,), name='digits')`
    `x = layers.Dense(64, activation='relu', name='dense_1')(inputs)`
    `x = layers.Dense(64, activation='relu', name='dense_2')(x)`
    `outputs = layers.Dense(10, activation='softmax', name='predictions')(x)`
    `model = keras.Model(inputs=inputs, outputs=outputs)`
    `return model`
  `model = get_uncompiled_model()`

  自定义损失函数，

**`def basic_loss_function(y_true, y_pred):`**
    **`return tf.math.reduce_mean(y_true - y_pred)`**

**model.compile会自动将预测结果和实际结果传入loss进行计算**

`model.compile(optimizer=keras.optimizers.Adam(),`
              **`loss=basic_loss_function`**

`)`
`model.fit(x_train, y_train, batch_size=64, epochs=3)`



# 验证数据集

**处理使用validation_data传入测试数据，还可以使用validation_split划分验证数据**

validation_split只能在用numpy数据训练的情况下使用

`validation_data=val_dataset`

`validation_split=0.2`

`steps_per_epoch 每个epoch只训练几步`

`validation_steps 每次验证，验证几步`

# 

# 使用回调
Keras中的回调是在训练期间（在epoch开始时，batch结束时，epoch结束时等）与不同时间点调用的对象，
可用于实现以下行为：

- 在训练期间的不同时间点进行验证（除了内置的按时间段验证）
- 定期检查模型是否超过某个精度阈值
- 在训练似乎停滞不前时，改变模型的学习率
- 在训练似乎停滞不前时，对顶层进行微调
- 在训练结束或超出某个性能阈值时发送电子邮件或即时消息通知等等。

**可使用的内置回调有**

- ModelCheckpoint：定期保存模型。
- EarlyStopping：当训练不再改进验证指标时停止培训。
- TensorBoard：定期编写可在TensorBoard中显示的模型日志（更多细节见“可视化”）。
- CSVLogger：将丢失和指标数据流式传输到CSV文件。



##  回调使用
`下面是几个回调使用的简单例子
<img src="E:\Typora\project\提前终止" alt="image-20210807153327688"  />

![image-20210807164938924](E:\Typora\project\模型保存)

![image-20210807165043602](E:\Typora\project\学习率调整)

![image-20210807165114478](E:\Typora\project\训练可视化)





# 构造梯度

![image-20210807165224034](E:\Typora\project\梯度计算)



在自定义训练中设定指标
我们可以在**自定义训练循环**中随时使用内置指标（或编写的自定义指标）。流程如下：

- 在循环开始时实例化指标
- 每一批数据训练之后，调用metric.update_state()
- 需要获得指标时，调用metric.result()
- 需要清除指标时，调用metric.reset_states()

![image-20210807165349096](E:\Typora\project\构造循环求梯度1)

![image-20210807165501725](E:\Typora\project\构造循环求梯度2)