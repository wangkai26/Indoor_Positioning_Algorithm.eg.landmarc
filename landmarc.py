import numpy as np
import matplotlib.pyplot as plt

# Calculate the Euclidean distance between two points
def dist(x,y):
    return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))


# Calculate the Euclidean distance between Tags and Readers
# eg. 121 Tags, 4 Readers,the matrix returned .shape = (4,121)
def mat_dist(tags,readers):
    n = len(readers)
    m = len(tags)
    ref_dist = [[0] * m for _ in range(n)]
    for i in range(len(readers)):
        for j in range(len(tags)):
            ref_dist[i][j] = dist(readers[i],tags[j])
    ref_dist = np.array(ref_dist)
    return ref_dist


# Calculate the Euclidean distance between the rssi value
# of the reference label and the test label
# eg. ref.shape=(4,121),test.shape=(4,5),the matrix returned .shape=(5,121)
def euclidean_distance(ref,test):
    m = test.shape[1] # 5
    n = ref.shape[1] # 121
    num_read = ref.shape[0]
    d = [[0] * n for i in range(m)]
    for i in range(m):
        for j in range(n):
            var = 0
            for k in range(num_read):
                var += np.square(ref[k][j] - test[k][i])
            d[i][j] = np.sqrt(var)
    return np.array(d)


# 选出 d 中每行前 k 个最大值，d.shape = (5,121)
# k 默认为 3
# the naming of be_knn is a bit random
# used to selected the top k largest values of each row,k defaults to 3
def be_knn(d,k=3):
    m,n = d.shape
    k = 3
    d = np.sort(d)
    sort_d = np.zeros((m,k))
    for i in range(m):
        sort_d[i] = d[i][-k:]
    return sort_d


# this knn is different to above be_knn
# knn is used to find the coordinates of the top k largest values in each row
def knn(d,k=3):
    m,n = d.shape
    k = 3
    d = np.argsort(d)
    sort_d = np.zeros((m,k))
    for i in range(m):
        sort_d[i] = d[i][:k]
    return sort_d


# Start positioning
# sort_d_cor,sort_d .shape both was(5,3),sort_d is rssi,sort_d_cor is coordinates
def predict(sort_d_cor,sort_d,ref_tag):
    m,k = sort_d_cor.shape
    res = np.zeros((m,2))
    for i in range(m):
        x,y = 0,0
        sum_w = 0
        for j in range(k):
            sum_w += 1/sort_d[i][j]
        w = [0] * k
        for j in range(k):
            w[j] = 1/sort_d[i][j] / sum_w
        for j in range(k):
            # print(ref_tag[int(sort_d_cor[i][j])])
            x += w[j] * ref_tag[int(sort_d_cor[i][j])][0]
            y += w[j] * ref_tag[int(sort_d_cor[i][j])][1]
        print(w)
        res[i] = x,y
    return res


# Calculate the RMSE of positioning
def pre_error(test,pred):
    m,n = len(test),len(pred)
    if m != n:
        print('demension error!')
    error = []
    for i in range(m):
        x1,y1 = test[i]
        x2,y2 = pred[i]
        error.append(np.sqrt(np.square(x1-x1))+np.square(y1-y2))
    system_error = sum(error)/m
    return error,system_error



# generate readers,There are four readers.
# Try to avoid integers, because the denominator cannot be 0 in division.
# The number and position coordinates can be changed according to your needs.
reader = [[0.5,0.5],[10.5,0.5],[0.5,10.5],[10.5,10.5]]
reader = np.array(reader)

# generate reference tag,total 121.
ref_tag = []
for i in range(11):
    for j in range(11):
        ref_tag.append([i,j])
ref_tag = np.array(ref_tag)

ref_read_dist = mat_dist(ref_tag,reader)

# ref_read_dist.shape = (4,121)
# generate the RSSI of reference Tags
# 对数路径损耗模型
n = 2.2  # 距离衰减因子
ref_rssi = -30 - 10 * n * np.log(ref_read_dist) #shape=(4,121)

# generate tag to be located
# generate the RSSI of test Tags
test_tag = np.array([[2,4],[5,5],[5,3],[7,7],[8,9]])
test_read_dist = mat_dist(test_tag,reader) # .shape=(4.5) no problem
test_rssi = -30 - 10 * n * np.log(test_read_dist) + np.random.randint(0,8,size=(4,5))#shape=(4,5)

D = euclidean_distance(ref_rssi,test_rssi) # (5,121)

sort_d = be_knn(D)
sort_d_cor = knn(D)   # .shape = (5,3)

# This two line codes is used for verification
# print(sord_d[0][0])
# print(ref_tag[26])

pred_coor = predict(sort_d_cor,sort_d,ref_tag)
print("result of predict",pred_coor)

# Calculate the RMSE
error,system_error = pre_error(test_tag,pred_coor)
print('mse:',error,system_error)



# x and y coordinates used for drawing
x_ref = [a[0] for a in ref_tag]
y_ref = [a[1] for a in ref_tag]
# x and y of readers
x_read = [a[0] for a in reader]
y_read = [a[1] for a in reader]

x_test = [a[0] for a in test_tag]
y_test = [a[1] for a in test_tag]

x_pred = [a[0] for a in pred_coor]
y_pred = [a[1] for a in pred_coor]
# plt.plot(xs[0],ys[0],"ro",[0,0,10,10],[0,10,0,10],'ro')
fig = plt.figure()
ax = fig.add_subplot(2,2,3)
ax.plot(x_ref,y_ref,"ro",x_read,y_read,'yo')

ax.plot(x_ref,y_ref,"ro",label='reference_tag')
ax.plot(x_read,y_read,'y^',label='reader')
ax.plot(x_test,y_test,'go',label='test_tag')
ax.plot(x_pred,y_pred,'bo',label='pred_tag')

for i in range(5):
    plt.arrow(x_test[i],y_test[i], x_pred[i]-x_test[i],y_pred[i]-y_test[i], ec='red', shape='full', length_includes_head='True', head_width=0.02)

# ax.legend(loc='best')
ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
# ax.legend(bbox_to_anchor=(1.25,1),loc='upper right')

name_list = [1,2,3,4,5,'system']
error.append(system_error)
ax2 = fig.add_subplot(2,2,2)
ax2.bar(x=range(len(error)),height=error,color='rgb',tick_label=name_list)
plt.show()
