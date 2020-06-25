# Media Eval Video Memorability
Prediction of Long-Term and Short-Term Video Memorability


##### This work focuses on how machine learning algorithms solve real-life problems and also introduced how efficiently marketing strategy can be done to grow up the sales of various goods and products. Video memorability also helpful in understanding the threats sent over the video to reduce the risks like terrorism, violence, and the spread of fake humor. Henceforth, this project focuses on how much a machine can remember a video by analyzing them for a long and short period. The various semantic features and video features have been available to test the experiment whereas I have focused on Semantic: Caption and Video: HMP and sum of 4 experiments which has been done in this work. The Machine learning techniques used in the experiments is Regression modeling to predict video memorability score for both terms. Finally, the best-scored model/experiment has been used for final modeling to generate good-fit data.

<center> Short-Term Spearman’s Correlation Coefficient Score.</center>

| Features  | Linear Regression | Decision Tree Regressor | Random Forest Regressor Est10 | Random Forest Regressor Est100
| ------------- | ------------- |  ------------- |  ------------- | ------------- |
| Weighted Caption  | 0.132 | 0.285 | 0.294 | 0.351 |
| Only Caption | 0.148  | 0.259| 0.289 | 0.325 |
| HMP | 0.043 | 0.087 | 0.224 | 0.321 |
| HMP + CAPTION | -0.008 | 0.093 | 0.224 | 0.318 |

<center> Long-Term Spearman’s Correlation Coefficient Score.</center>

| Features  | Linear Regression | Decision Tree Regressor | Random Forest Regressor Est10 | Random Forest Regressor Est100
| ------------- | ------------- |  ------------- |  ------------- | ------------- |
| Weighted Caption  | 0.030 | 0.093 | 0.158 | 0.193 |
| Only Caption | 0.036  | 0.127| 0.148 | 0.154 |
| HMP | -0.027 | 0.043 | 0.056 | 0.112 |
| HMP + CAPTION |0.038 | 0.038 | 0.035 | 0.138 |


# Conclusion
##### In this research, the work described the robust way to compute the memorability, the finding is the semantic feature is more capable and accurate to do a prediction for long-term and short- term video memorability score where addition support to word bag corpus leads result to outstanding spearman’s score performance, thus found that weighted caption has the highest and significant score and this implies that caption is a good source to do furthermore investigation in the domain of video memorability.


To learn more, Kindly go through documentation <b>"Report.pdf"</b>.

###### imports used
* import pandas as pd
* import numpy as np
* import seaborn as sns
* import matplotlib.pyplot as plt
* import plotly.express as px
* from sklearn.model_selection import  train_test_split,cross_val_score
* from sklearn.preprocessing import StandardScaler
* from sklearn.linear_model import LinearRegression
* from sklearn import metrics
* from sklearn.metrics import r2_score
* from sklearn.ensemble import RandomForestRegressor
* from sklearn.ensemble import GradientBoostingRegressor
* from sklearn.model_selection import ShuffleSplit
* from sklearn.svm import SVR
* from sklearn.tree import DecisionTreeRegressor
