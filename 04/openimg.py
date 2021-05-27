import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# 이미지 데이터를 Average Hash로 변환하기 --- (※1)
def binary_level(fname, size = 16):
    img = Image.open(fname) # 이미지 데이터 열기---(※2)
    img = img.convert('L') # 그레이스케일로 변환하기 --- (※3)
    img = img.resize((size, size), Image.ANTIALIAS) # 리사이즈하기 --- (※4)
    pixel_data = img.getdata() # 픽셀 데이터 가져오기 --- (※5)
    pixels = np.array(pixel_data) # Numpy 배열로 변환하기 --- (※6)
    pixels = pixels.reshape((size, size)) # 2차원 배열로 변환하기 --- (※7)
    avg = pixels.mean() # 평균 구하기 --- (※8)
    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로 변환하기 --- (※9)
    return diff

def eight_bit_level_file(fname, size = 16):
    img = Image.open(fname) # 이미지 데이터 열기---(※2)
    img = img.convert('L') # 그레이스케일로 변환하기 --- (※3)
    img = img.resize((size, size), Image.ANTIALIAS) # 리사이즈하기 --- (※4)
    pixel_data = img.getdata() # 픽셀 데이터 가져오기 --- (※5)
    pixels = np.array(pixel_data) # Numpy 배열로 변환하기 --- (※6)
    pixels = pixels.reshape((size, size)) # 2차원 배열로 변환하기 --- (※7)
#    avg = pixels.mean() # 평균 구하기 --- (※8)
#    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로 변환하기 --- (※9)
    return pixels

def eight_bit_level_data(img, size = 16):
#    img = Image.open(fname) # 이미지 데이터 열기---(※2)
#    img = img.convert('L') # 그레이스케일로 변환하기 --- (※3)
    img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA) 
#    print(img)
#    img = img.resize((size, size), Image.ANTIALIAS) # 리사이즈하기 --- (※4)
#    pixel_data = img.getdata() # 픽셀 데이터 가져오기 --- (※5)
#    pixels = np.array(pixel_data) # Numpy 배열로 변환하기 --- (※6)
#    pixels = pixels.reshape((size, size)) # 2차원 배열로 변환하기 --- (※7)
#    avg = pixels.mean() # 평균 구하기 --- (※8)
#    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로 변환하기 --- (※9)
    return img


#이미지 불러오기
gray_image = cv2.imread('D:\\py_lab\\py_lec\\03\\fish_input.jpg', 0) 
rgb_image = cv2.imread('D:\\py_lab\\py_lec\\03\\fish_input.jpg')

im = plt.imread('D:\\py_lab\\py_lec\\03\\fish_input.jpg')

#이미지 보기
#cv2.imshow('gray_image', gray_image)
#cv2.imshow('rgb_image', rgb_image)


print(np.shape(gray_image))

height, width = np.shape(gray_image)

# x = np.arange(0, width)
# y = gray_image[int(height/2), 0:width]

# implot = plt.imshow(im, cmap='gray')

# plt.plot(x, y)
# plt.show()

ahash = binary_level('fish_input.jpg')
hash = eight_bit_level_file('fish_input.jpg')

print(hash)
print(ahash)


img_b,img_g,img_r = cv2.split(rgb_image)



hash_r = eight_bit_level_data(img_r)
hash_g = eight_bit_level_data(img_g)
hash_b = eight_bit_level_data(img_b)

print(hash_r)
print(hash_g)
print(hash_b)

zeros = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype="uint8")
img_b = cv2.merge([img_b, zeros, zeros])
img_g = cv2.merge([zeros, img_g, zeros])
img_r = cv2.merge([zeros, zeros, img_r])

cv2.imshow("BGR", rgb_image)
cv2.imshow("B", img_b)
cv2.imshow("G", img_g)
cv2.imshow("R", img_r)



cv2.waitKey(0)

cv2.destroyAllWindows()
