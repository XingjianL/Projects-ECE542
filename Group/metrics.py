import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

results = pd.read_csv("/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/Models/resnet50transfer_preds.csv",
                      header=0)
labels = results.iloc[:,8].to_numpy().flatten()
# get recorded model outputs
non_softmax_scores = results.iloc[:,:7].to_numpy()
# convert to probabilites
softmax_scores = np.exp(non_softmax_scores) / np.sum(np.exp(non_softmax_scores), axis=1).reshape(-1,1)
# probability of the true label
label_scores = [softmax_scores[i,labels[i]] for i in range(len(labels))]

print(f"avg_roc_auc_score_weighted: {metrics.roc_auc_score(labels, softmax_scores, multi_class='ovr', average='weighted'):.4f}")
print(f"avg_roc_auc_score_unweighted: {metrics.roc_auc_score(labels, softmax_scores, multi_class='ovr', average='macro'):.4f}")


name = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
fig, axes = plt.subplots(3,3)
figorder = []
for i in range(7):
    binary_label = [1 if labels[j] == i else 0 for j in range(len(labels))]
    binary_score = softmax_scores[:, i]

    #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{name[i]} vs. rest Classifier")
    display = metrics.RocCurveDisplay.from_predictions(binary_label, binary_score, name=f"{name[i]} vs. rest Classifier", ax=axes[int(i/3), i%3])
    axes[int(i/3), i%3].plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")


preds = results.iloc[:,7].to_numpy().flatten()
precision, recall, fscore, support = metrics.precision_recall_fscore_support(labels, preds)

print(f"avg_precision: {np.mean(precision):.4f} \navg_recall: {np.mean(recall):.4f}")

plt.show()