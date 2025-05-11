

def precision(y_pred, y_real):
    pred_1 = y_pred.count(1.)
    real_1 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==1])
    if pred_1 ==0:
        return 0
    return real_1/pred_1

def recall(y_pred, y_real):
    real_1 = y_real.count(1.)
    pred_1 = sum([1 for x, y in zip(y_pred, y_real) if x==y and x==1])
    return pred_1/real_1

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