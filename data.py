import statistics as st
data = [15,101,18,7,13,16,11,21,5,15,10,9]
print('WITH OUTLIER')
print('mean',st.mean(data))
print('median',st.median(data))
print('mode',st.mode(data))
print('variance',st.variance(data))
print('std dev',st.stdev(data))
### Outlier z-score
import numpy as np
def detect_outlier_zscore(data):
threshold = 3
mean = np.mean(data)
std = np.std(data)
outliers = []
for i in data:
z_score = (i-mean)/std
if(np.abs(z_score)>threshold):
outliers.append(i)

return outliers
out = detect_outlier_zscore(data)
print('outlier:',out)
newData = [i for i in data if i not in out]
print('\t\tWITH OUTLIER \tWITHOUT OUTLIER')
print('mean\t\t',round(st.mean(data),2),'\t\t',round(st.mean(newData),
2))
print('median\t\t',round(st.median(data),2),'\t\t',round(st.median(new
Data),2))
print('mode\t\t',round(st.mode(data),2),'\t\t',round(st.mode(newData),
2))
print('variance\t',round(st.variance(data),2),'\t',round(st.variance(n
ewData),2))
print('std
dev\t\t',round(st.stdev(data),2),'\t\t',round(st.stdev(newData),2))

### Outlier iqr
def detect_outliers_iqr(data):
outliers =[]
data = sorted(data)
q1 = np.percentile(data,25)
q3 = np.percentile(data,75)
iqr = q3 - q1
lwr_bound = q1-(1.5*iqr)
upr_bound = q3+(1.5*iqr)
for i in data:
if(i<lwr_bound or i>upr_bound):
outliers.append(i)

return outliers
sample_outliers = detect_outliers_iqr(data)
print('Outliers:',sample_outliers)
### quantile based flooring and capping
tenth_percentile = np.percentile(data,10)
ninetieth_percentile = np.percentile(data,90)
b = np.where(data<tenth_percentile,tenth_percentile,data)
b = np.where(data>ninetieth_percentile,ninetieth_percentile,b)
print('10%',tenth_percentile,'\n90%',ninetieth_percentile,'\nNew
Array:',b)
