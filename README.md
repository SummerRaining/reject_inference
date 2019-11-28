### 拒绝推断的代码和部分结果
1. roc曲线图像在roc_image文件中，其中ROC curve based on the accepted.png是四种图像画在一张图上的结果。
2. 接受样本上，每个模型的准确率，第一类第二类错误率等性能结果在result_log.txt文件中。
3. 并入拒绝样本后，训练得到的各项指标结果在reject_log.txt文件中。
4. reject_probas.png 是stacking模型对拒绝样本预测概率的直方图。可以看出其预测结果分布在0.1左右，所以使用0.5作为阈值，几乎所有样本都被划分到标签0了。
![reject_probas](./reject_probas.png)
