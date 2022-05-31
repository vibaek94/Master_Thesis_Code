import pandas as pd

from data_prep import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


wb = openpyxl.load_workbook(r'C:\Users\jcb\Desktop\speciale/ChildrenThighSimple10.xlsx', read_only=True)
sheet = wb.worksheets[0]
ws = wb.active

data = ws.values
cols = next(data)[1:]
data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 1, None) for r in data)

df1 = pd.DataFrame(data, index=idx, columns=cols)
df1['ID'] = df1.index
REE, Weight, id, correction = list(), list(), list(), list()
for x in range(df1['ID'].min(), df1['ID'].max() + 1, 1):
    if (x in df1['ID']):
        temp = df1.loc[lambda df1: df1['ID'] == x]
        temp1 = temp.loc[lambda temp: temp['ActivityType'] < 4]
        corr = temp['vo2'] - ((temp['vo2n']) * 50 * temp1['Weight'].mean() * 100 + temp1['vo2'].mean())
        correction.append(corr.mean())
        REE.append(temp1['vo2'].mean())
        Weight.append(temp1['Weight'].mean() * 100)
        id.append(x)
feat = pd.DataFrame(list(zip(REE, Weight, correction)),
                    columns=['Resting EE', 'Weight', 'Correction'], index=id)

X = df1.iloc[1:-1,[3,1,7,8,9]]
VO2 = df1.iloc[2:,0]
ID = df1.iloc[2:,13]
y = df1.iloc[2:,11]
X_train_lr=X.iloc[9:7498]
print(X_train_lr.shape)
X_test_lr=X.iloc[(9041):]
print(X_test_lr.shape)
y_train_lr=y.iloc[9:7498]
print(y_train_lr.shape)
y_test_lr= y.iloc[(9041):]
print(y_test_lr.shape)
VO2_test = VO2.iloc[(9041):]
id_test = ID.iloc[(9041):]


lr = LinearRegression()
model = lr.fit(X_train_lr, y_train_lr)
yhat_lr = model.predict(X_test_lr)
yhat_lr = pd.Series(yhat_lr)
mse = mean_squared_error(y_test_lr, yhat_lr)
mape = mean_absolute_percentage_error(y_test_lr, yhat_lr)
print('MSE: %.3f MAPE: %.3f%%' % (mse, mape))
print('R^2: %.3f' % r2_score(y_test_lr, yhat_lr))
print('RMSE: %.3f'%mean_squared_error(y_test_lr, yhat_lr,squared=False))
#model_name="LinearRegression_Model"
#parent_dir = r'C:\Users\jcb\Desktop\speciale\models'
# Path
#path = os.path.join(parent_dir, model_name)
# makes directory for the model name

# plot of the values for predicted y and actual y. also plots R^2, MSE and MAPE
mse = mean_squared_error(y_test_lr, yhat_lr)
mape = mean_absolute_percentage_error(y_test_lr, yhat_lr)
R2 = r2_score(y_test_lr, yhat_lr)
y_test_lr_NP = y_test_lr.to_numpy()
yhat_lr_NP = yhat_lr.to_numpy()
yhat_lr_VO2 = calc_B2_VO2(feat, yhat_lr_NP, y_test_lr_NP, VO2_test, id_test)
yhat_lr_VO2 = yhat_lr_VO2.to_numpy()
yhat_lr_VO2 = pd.Series(yhat_lr_VO2)
VO2_test = VO2_test.to_numpy()
pred_error_lr = yhat_lr_NP - y_test_lr_NP
pred_error_lr_VO2 = yhat_lr_VO2 - VO2_test
pred_error_lr_VO2 = pd.Series(np.absolute(pred_error_lr_VO2))
pred_error_lr = np.absolute(pred_error_lr)
pred_error_lr = pd.Series(pred_error_lr)

"""
print('MSE: %.3f MAPE: %.3f%%' % (mse, mape))
plt.style.use('ggplot')
plt.figure(11)
plt.plot(yhat_lr, label="Predicted y")
plt.plot(y_test_lr.to_numpy(), label="Actual y")
plt.legend(['R^2: %.3f' % r2_score(y_test_lr, yhat_lr), 'MSE: %.3f MAPE: %.3f%%' % (mse, mape)], loc='lower right')
print(path)
plt.show()
plt.savefig(path + '\\' + model_name + '_fit.png')
plt.close()"""

mse = mean_squared_error(VO2_test, yhat_lr_VO2)
mape = mean_absolute_percentage_error(VO2_test, yhat_lr_VO2)

model_name = 'Linear_regression_MAD'
parent_dir = r'C:\Users\jcb\Desktop\speciale\Results'
path = os.path.join(parent_dir, model_name)
if os.path.isdir(path) == False:
    os.mkdir(path)
pred_error_lr.to_csv(path + "/Result_mean_absolute_error",index=False)
yhat_lr.to_csv(path + "/Result_mean_prediction",index=False)

model_name = 'Linear_regression_VO2_MAD'
parent_dir = r'C:\Users\jcb\Desktop\speciale\Results'
path = os.path.join(parent_dir, model_name)
if os.path.isdir(path) == False:
    os.mkdir(path)
pred_error_lr_VO2.to_csv(path + "/Result_mean_absolute_error_VO2",index=False)
yhat_lr_VO2.to_csv(path + "/Result_mean_prediction_VO2",index=False)

dict_metric = {}
dict_metric["MAPE"]=[mape]
dict_metric["R2"]=[R2]

file = open(path + "/Metrics_Thigh", "w")
file.write("%s = %s\n" % ("Metrics", dict_metric))
file.close()



