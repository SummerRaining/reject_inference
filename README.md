### 拒绝推断的代码和部分结果
1. roc曲线图像在roc_image文件中，其中ROC curve based on the accepted.png是四种图像画在一张图上的结果。
2. 接受样本上，每个模型的准确率，第一类第二类错误率等性能结果在result_log.txt文件中。
3. 并入拒绝样本后，训练得到的各项指标结果在reject_log.txt文件中。
4. reject_probas.png 是stacking模型对拒绝样本预测概率的直方图。可以看出其预测结果分布在0.1左右，所以使用0.5作为阈值，几乎所有样本都被划分到标签0了。
![reject_probas](./reject_probas.png)


### type I error rate的定义
1. 阅读参考论文得到第一类错误率的计算公式。
	- 第一类错误率 = FN/(FN+TP)。第二类错误率 = FP/(FP+TN)。
	- 可以看出FN+TP是实际上所有的正样本数，recall = TN/(TN+FP)。因此第一类错误率 = 1-recall。
	- FP+TN是实际上所有的负样本数，第二类错误率无明显关系。

