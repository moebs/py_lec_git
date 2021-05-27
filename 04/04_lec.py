import datetime 
date_time = datetime.datetime.now() 
print(date_time)

date_today = datetime.date.today() 
print(date_today)

#####################################

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 2*3.141592, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

########################################
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection='3d')
ax.scatter(np.random.rand(5),np.random.rand(5),np.random.rand(5))
plt.show()
######################################

days=['(월)','(화)','(수)','(목)','(금)','(토)','(일)']

######################################

# pip install opencv-python
# True  = 1
# False = 0

import cv2

cap = cv2.VideoCapture('D:\\py_lab\\py_lec\\03\\face5.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    print(ret, frame)
    print(type(ret))
    if(ret):
        cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == 27: # press esc-key to exit
        break

cap.release()
cv2.destroyAllWindows()



# for i in range(0, 239+1):
#     ret, frame = cap.read()
#     print(i, ret)
#     if(ret):

#         cv2.imshow('frame', frame)
#     if cv2.waitKey(5) & 0xFF == 27: # press esc-key to exit
#         break
# cap.release()
# cv2.destroyAllWindows()


##############################################



#score = [ 72, 85, 52, 92, 64 ]   

i = 75

if ( 90 <= i <=100):
    print('A')
elif ( 80 <= i < 90):
    print('B')
elif ( 70 <= i < 80):
    print('C')
elif ( 60 <= i < 70):
    print('D')
else :
    print('F')

#####################################

for a in range( 2 , 9+1):
    for b in range( 1 , 9+1):
        print( a,'x',b,'=',a*b)
    print('\n')


########################################

tree_hit = 1
while tree_hit <=10:
    print('나무 찍은 횟수 : ', tree_hit)
    if tree_hit == 10:
        print('나무 넘어갑니다')
    tree_hit = tree_hit +1    

