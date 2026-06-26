
# 1 说明
这个分支实现了论文里的深度学习算法，复现结果保存如下

## 1.1 GitHub Command
远程提交
```bash
git push -u AFCI-github CNN-rep
```


拉取远程最新更改
```bash
git fetch --all
```


强制覆盖本地分支
```bash
git reset --hard origin/master
```


## 1.2 训练模型
```python
CUDA_VISIBLE_DEVICES=0 nohup python main.py > output.log 2>&1 &
```
CUDA_VISIBLE_DEVICES=0 nohup python main.py > output_LR_single.log 2>&1 &
# 2 数据集说明
```python
''' 10个低频, 15个高频
00-10000_multi_airconditioner02-refrigerator_m2.yaml
01-10000_multi_kettle01-microwaveoven_m2.yaml
02-10000_multi_refrigerator-airconditioner_m2.yaml
03-10000_multi_ricecooker-electricoven_m2.yaml
04-10000_multi_inductioncooker-airpurifier_m2.yaml
05-10000_multi_airpurifier-airconditioner03_m2.yaml
06-10000_multi_microwaveoven-waterheater_m2.yaml
07-10000_multi_electricoven-washingmachine_m2.yaml
08-10000_multi_airconditioner01-inductioncooker_m2.yaml
09-10000_single_inductioncooker_m1.yaml
10-10000_single_refrigerator_m1.yaml
11-10000_single_electricoven_m1.yaml
12-10000_single_ricecooker_m1.yaml
13-10000_single_kettle01_m1.yaml
14-10000_single_microwaveoven_m1.yaml

15-6400_multi_microwaveoven-refrigerator_m3.yaml
16-6400_multi_electricoven-refrigerator_m3.yaml
17-6400_multi_kettle02-airpurifier_m3.yaml
18-6400_single_inductioncooker_m1.yaml
19-6400_single_refrigerator_m1.yaml
20-6400_single_kettle02_m1.yaml
21-6400_single_electricoven_m1.yaml
22-6400_single_ricecooker_m1.yaml
23-6400_single_kettle01_m1.yaml
24-6400_single_microwaveoven_m1.yaml
'''
```

## 2.1 Confusion Matrix
| Actual \ Predicted | Positive | Negative |
|--------------------|----------|----------|
| **Positive**       | TP       | FN       |
| **Negative**       | FP       | TN       |

## Evaluation Metrics

### Basic Metrics
| Metric      | Formula                          | Description                          |
|-------------|----------------------------------|--------------------------------------|
| **FPR**     | FP / (FP + TN)                   | False Positive Rate (Type I Error)   |
| **FNR**     | FN / (FN + TP)                   | False Negative Rate (Type II Error)  |
| **Precision**| TP / (TP + FP)                   | Positive Predictive Value            |
| **Recall**  | TP / (TP + FN)                   | Sensitivity, True Positive Rate      |
| **Accuracy**| (TP + TN) / (TP + TN + FP + FN)  | Overall Correctness                  |

### F-Scores
| Metric | Formula                                                  | Weight Emphasis       |
|--------|----------------------------------------------------------|-----------------------|
| **F1** | 2 × (Precision × Recall) / (Precision + Recall)          | Balanced (β=1)        |
| **F2** | (1+2²) × (Precision × Recall) / (2²×Precision + Recall) | Recall-focused (β=2) |
| **F0.5**| (1+0.5²) × (Precision × Recall) / (0.5²×Precision + Recall)| Precision-focused (β=0.5) |



# 传统机器学习实验结果
## HR-mix1
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.021877       0.080023       0.062752  0.014393             0.276339     0.027058           0.010075
FNR        0.120553       0.204545       0.114625  0.162055             0.306324     0.791502           0.142292
Precision  0.921325       0.743306       0.804309  0.944321             0.422383     0.691803           0.961240
Recall     0.879447       0.795455       0.885375  0.837945             0.693676     0.208498           0.857708
Accuracy   0.955863       0.891886       0.925546  0.952296             0.716897     0.800490           0.960098
F1         0.899899       0.768496       0.842897  0.887958             0.525056     0.320425           0.906527
F2         0.887515       0.784447       0.867881  0.857258             0.614711     0.242362           0.876591
F0.5       0.912633       0.753181       0.819312  0.920938             0.458225     0.472670           0.938581

## HR-single
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.005528       0.009828       0.007371  0.020885             0.062039     0.003686           0.009214
FNR        0.132743       0.125369       0.119469  0.106195             0.197640     0.738938           0.078171
Precision  0.984925       0.973727       0.980296  0.946875             0.843411     0.967213           0.976562
Recall     0.867257       0.874631       0.880531  0.893805             0.802360     0.261062           0.921829
Accuracy   0.957069       0.956201       0.959670  0.954033             0.898092     0.780139           0.970512
F1         0.922353       0.921523       0.927739  0.919575             0.822373     0.411150           0.948407
F2         0.888486       0.892803       0.898826  0.903938             0.810247     0.305699           0.932279
F0.5       0.958904       0.952152       0.958574  0.935763             0.834868     0.627660           0.965102

## LR-mix2
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.030588       0.042353       0.028235  0.023529             0.034118     0.025882           0.020000
FNR        0.169118       0.144608       0.129902  0.186275             0.259804     0.571078           0.142157
Precision  0.928767       0.906494       0.936675  0.943182             0.912387     0.888325           0.953678
Recall     0.830882       0.855392       0.870098  0.813725             0.740196     0.428922           0.857843
Accuracy   0.924483       0.924483       0.938792  0.923688             0.892687     0.797297           0.940382
F1         0.877102       0.880202       0.902160  0.873684             0.817321     0.578512           0.903226
F2         0.848773       0.865146       0.882645  0.836694             0.769231     0.478403           0.875438
F0.5       0.907388       0.895791       0.922557  0.914097             0.871824     0.731605           0.932836


## LR-single
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.042778       0.042222       0.043889  0.046667             0.077778     1.000000           0.030556
FNR        0.225172       0.219272       0.196657  0.198623             0.288102     0.018682           0.195674
Precision  0.910983       0.912644       0.911830  0.906563             0.837963     0.356683           0.936999
Recall     0.774828       0.780728       0.803343  0.801377             0.711898     0.981318           0.804326
Accuracy   0.891374       0.893859       0.900958  0.898474             0.846290     0.354278           0.909833
F1         0.837407       0.841547       0.854156  0.850731             0.769803     0.523198           0.865608
F2         0.798703       0.803969       0.822925  0.820415             0.733982     0.726770           0.827768
F0.5       0.880054       0.882811       0.887850  0.883373             0.809300     0.408715           0.907075

## HR-all
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.014532       0.043015       0.019182  0.014338             0.249758     0.018989           0.009688
FNR        0.153280       0.173513       0.151441  0.148988             0.334151     0.859595           0.170448
Precision  0.948489       0.858599       0.933243  0.949384             0.457263     0.700306           0.964362
Recall     0.846720       0.826487       0.848559  0.851012             0.665849     0.140405           0.829552
Accuracy   0.952150       0.925648       0.949058  0.953327             0.729976     0.779152           0.951708
F1         0.894720       0.842237       0.888889  0.897511             0.542187     0.233912           0.891892
F2         0.865288       0.832716       0.864244  0.869021             0.610181     0.167129           0.853412
F0.5       0.926224       0.851978       0.914981  0.927932             0.487827     0.389588           0.934005

## LR-all
                SVM  Decision Tree  Random Forest       KNN  Logistic Regression  Naive Bayes  Gradient Boosting
FPR        0.040931       0.049944       0.049568  0.046564             0.072850     0.999624           0.033421
FNR        0.211756       0.190510       0.174929  0.188385             0.260623     0.013456           0.179178
Precision  0.910802       0.895768       0.898227  0.902362             0.843296     0.343527           0.928686
Recall     0.788244       0.809490       0.825071  0.811615             0.739377     0.986544           0.820822
Accuracy   0.899877       0.901350       0.906994  0.904294             0.862086     0.342086           0.916074
F1         0.845103       0.850446       0.860096  0.854586             0.787925     0.509603           0.871429
F2         0.810044       0.825390       0.838733  0.828274             0.758060     0.717819           0.840342
F0.5       0.883333       0.877072       0.882576  0.882625             0.820239     0.395020           0.904903

# CNN实验结果