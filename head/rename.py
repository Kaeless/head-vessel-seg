import os
import sys

path = ('./images/')

#获取该目录下所有文件，存入列表中
f =os.listdir(path)
print(len(f))

#删除.jpg后缀,只要文件名前13位数字
newarr = []
for i in f:
    x = i[0:8]
    newarr.append(x)
    
#文件名从大到小排序
newarr.sort()



n = 1

for i in range(len(newarr)):
    # oldname = path + newarr[i] +'.jpg'
    oldname = path + newarr[i]
    newname = "%03d.png"%n
    n = n + 1
    #第0,2,4...偶数位是反面,命名+b , 奇数位是正面,命名+a
    # if i % 2 == 0 :
    #     newname = path + str(n) + 'b' + '.jpg'
    # else:
    #     newname = path + str(n) + 'a' + '.jpg'
    #     n = n + 1

    #用os模块中的rename方法对文件改名
    os.rename(oldname,newname)
    print(oldname,'======>',newname)
    
    
