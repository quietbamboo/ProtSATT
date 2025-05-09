

# 预测的结果中，正确预测出1的比例
def precision(y_pred, y_real):
    # 预测为1的个数
    pred_1 = y_pred.count(1.)
    # 真实为1并预测也为1的个数
    real_1 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==1])
    if pred_1 ==0:
        return 0
    return real_1/pred_1

# 在真实结果中，有多少1被正确预测出来
def recall(y_pred, y_real):
    # 真实为1的个数
    real_1 = y_real.count(1.)
    # 真实为1并且被预测也为1的个数
    pred_1 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==1])
    return pred_1/real_1

# 准确率：被正确预测出来的样本比例
def accuracy(y_pred, y_real):
    correct = sum([1 for x, y in zip(y_pred, y_real) if x==y])
    return correct/len(y_real)

def f1_score(y_pred, y_real):
    pre = precision(y_pred, y_real)
    rec = recall(y_pred, y_real)
    if pre+rec == 0:
        return 0
    return 2*pre*rec/(pre+rec)

def detail(y_pred, y_real):
    correct_0 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==0])
    correct_1 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==1])
    print("0===========", correct_0,"/",y_real.count(0.), correct_0/y_real.count(0.))
    print("1===========", correct_1,"/",y_real.count(1.), correct_1/y_real.count(1.))