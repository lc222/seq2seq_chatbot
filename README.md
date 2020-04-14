# seq2seq_chatbot

=================================================分割线，下面是正文===============================================

本文是一个简单的基于 seq2seq 模型的 chatbot 对话系统的 tensorflow 实现。

代码的讲解可以参考我的知乎专栏文章：

[从头实现深度学习的对话系统--简单 chatbot 代码实现](https://zhuanlan.zhihu.com/p/32455898)

代码参考了 DeepQA，在其基础上添加了 beam search 的功能和 attention 的机制，

最终的效果如下图所示：

![](https://i.imgur.com/pN7AfAB.png)

![](https://i.imgur.com/RnvBDwO.png)

测试效果，根据用户输入回复概率最大的前 beam_size 个句子：

![](https://i.imgur.com/EdsQ5FE.png)

#使用方法

1，下载代码到本地（data 文件夹下已经包含了处理好的数据集，所以无需额外下载数据集）

2，训练模型，将 chatbot.py 文件第 34 行的 decode 参数修改为 False，进行训练模型

（之后我会把我这里训练好的模型上传到网上方便大家使用）

3，训练完之后（大概要一天左右的时间，30 个 epoches），再将 decode 参数修改为 True

就可以进行测试了。输入你想问的话看他回复什么吧==

这里还需要注意的就是要记得修改数据集和最后模型文件的绝对路径，不然可能会报错。

分别在 44 行，57 行，82 行三处。好了，接下来就可以愉快的玩耍了~~
