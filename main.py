import math
import pandas as pd
import numpy as np
import random
df = pd.read_csv('D:\可爱臭鸭鸭\duck_data.csv')#需要添加目标文件目录
property=['color','tail','sound','fur','belly','web','food','health']
train_data = np.array(df.iloc[:,1:10])
#============归一化=================
food=train_data[:,6]
health=train_data[:,7]
food_min=np.min(food)
food_max=np.max(food)
health_min=np.min(health)
health_max=np.max(health)
for i in range(len(food)):
    food[i]=( food[i]-food_min)/(food_max-food_min)
    food[i]=round( food[i],2)
for i in range(len(health)):
    health[i] = (health[i]- health_min) / (health_max - health_min)
    health[i] = round(health[i],2)
#============提取去除标签=================
row_rand_array = np.arange(train_data.shape[0])
np.random.shuffle(row_rand_array)
row_rand = train_data[row_rand_array[0:10]]#抽取12条数据训练
rest_rand = train_data[row_rand_array[10:17]]#抽取12条数据训练
train_data=row_rand
test_data=rest_rand
test_label=test_data[:,8]
train_label=train_data[:,8]
train_data=np.delete(train_data,8,1)#第三个数1表示列，0表示行
test_data=np.delete(test_data,8,1)#第三个数1表示列，0表示行
m,n=np.shape(train_data)
c,d=np.shape(test_data)
#得到先验概率和好鸭坏鸭个数
def get_pc_pnum(train_label):
  num0=0
  num1=0
  for j in train_label:
      if j=='no':
          num0+=1
      else:
          num1+=1
  Pc_num={'yes':num0,'no':num1}
  num0=(num0+1)/(len(train_label)+2)
  num1=(num1+1)/(len(train_label)+2)
  Pc={'yes':num0,'no':num1}
  return Pc,Pc_num
def get_total_p(train_data,train_label,Pc_num):
  #离散========================================
  total_p={}
  for j in range(0,6):
    value=[]
    for i in train_data[:,j]:
        if i not in value:
            value.append(i)
    mid_p={}
    for va in value:
        mid_p[va]={}


    for va in value:
        num0 = 0
        num1 = 0
        for i in range(len(train_data[:,j])):
            if train_label[i]=='yes':
               if va==train_data[i,j]:
                   num1+=1
            else:
                if va == train_data[i,j]:
                    num0 += 1

        mid_p[va]['yes']=np.round((num1+1)/(Pc_num['yes']+len(value)),4)
        mid_p[va]['no'] = np.round((num0+1)/(Pc_num['no']+len(value)),4)
    total_p[property[j]]=mid_p
  return total_p
def get_u_theta(train_data, train_label):
  #连续========================================
  u_theta={}
  for j in range(6,8):
    yes_list =[]
    no_list = []
    u=0
    theta=0
    for i in range(len(train_label)):
       if train_label[i]=='yes':
           yes_list.append(train_data[i,j])
       else:
           no_list.append(train_data[i,j])
    # ===========================
    u_theta[property[j]]={}
    u=np.sum(yes_list)/len(yes_list)
    sum=0
    for yes in yes_list:
        sum+=(yes-u)**2
    sum=sum/len(yes_list)
    u_theta[property[j]]['yes']=[np.round(u,3),np.round(sum,3)]
    #===========================
    u=np.sum(no_list)/len(no_list)
    sum=0
    for no in no_list:
        sum+=(no-u)**2
    sum=sum/len(no_list)
    u_theta[property[j]]['no']=[np.round(u,3),np.round(sum,3)]
  return u_theta
def get_result(test_data):
  result=[]
  Pc,Pc_num=get_pc_pnum(train_label)
  total_p=get_total_p(train_data,train_label,Pc_num)
  u_theta=get_u_theta(train_data, train_label)
  for i in range(len(test_data)):
    p0=1
    p1=1
    for j in range(0,6):
        p0 *= total_p[property[j]][test_data[i, j]]['no']
        p1 *= total_p[property[j]][test_data[i, j]]['yes']
    for j in range(6,8):
        u_yes     =u_theta[property[j]]['yes'][0]
        theta_yes =u_theta[property[j]]['yes'][1]
        u_no     =u_theta[property[j]]['no'][0]
        theta_no =u_theta[property[j]]['no'][1]
        p_yes=1/(np.sqrt(2*3.1415*theta_yes))*np.exp(-(test_data[i,j]-u_yes)**2/theta_yes)
        p_no = 1 / (np.sqrt(2 * 3.1415 * theta_no)) * np.exp(-(test_data[i, j] - u_no) ** 2 / theta_no)
        p0 *= p_no
        p1 *= p_yes
        haha=0

    if p0>p1:
        duck='no'
    else: duck='yes'
    result.append(duck)
  return result
def get_accuracy(test_data):
  result=get_result(test_data)
  num=0
  for i in range(len(test_label)):
    if test_label[i]!=result[i]:
        num+=1
  num=np.round((1-num/len(test_label))*100,2)
  return num



accuracy=get_accuracy(test_data)
print(('正确率为：'+str(accuracy)+'%'))
