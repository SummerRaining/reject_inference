### 拒绝推断的代码和部分结果
1. roc曲线图像在roc_image文件中，其中ROC curve based on the accepted.png是四种图像画在一张图上的结果。
2. 接受样本上，每个模型的准确率，第一类第二类错误率等性能结果在result_log.txt文件中。
3. 并入拒绝样本后，训练得到的各项指标结果在reject_log.txt文件中。
4. reject_probas.png 是stacking模型对拒绝样本预测概率的直方图。可以看出其预测结果分布在0.1左右，所以使用0.5作为阈值，几乎所有样本都被划分到标签0了。
![reject_probas](./reject_probas.png)

5. 混淆矩阵在reject_log.txt,accept_log.txt,hard_reject_log.txt文件中有记载。
	1. 例如TP:81，TN:3705,FN:349,FP:165.即代表true positive真阳，true negetive 真阴，false negative 假阴，false positive假阳

6. parameter.txt记载了树形模型的模型参数和调参后的结果。





