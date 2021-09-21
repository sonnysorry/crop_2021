import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 디렉토리를 설정합니다.
print(os.getcwd())
dir = os.path.join(tmp_dir, "images")

# 레이블을 지정합니다. (상단에 설정한 dir path 내 폴 명을더 아래의 이름으로 설정합니다.)
categories = []
while True:
    check_name = str(input("images 폴더 내 작물 폴더명을 정확하게 적으세요.(다 적고 0 입력 하면 시작합니다.) "))
    if check_name == "0":
        break
    categories.append(check_name)

data = []

# 각 카테고리를 보고 path의 하위 디렉토리로 설정하여 이미지를 뽑아옵니다.
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        crop_img = cv2.imread(imgpath, 0)
        try:
            crop_img = cv2.resize(crop_img, (500,500))
            image = np.array(crop_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass

#print(len(data))
# 데이터 변환 코드입니다. pickle(2진수 형태)로 변환하여
# 학습에 용이하게 사용합니다.
pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

# 변환한 pickle 파일을 로드하는 코드입니다.
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

# 임의 추출하는 코드입니다. 테스트를 위해 사용합니다.
random.shuffle(data)
features = []
labels = []

# 위에 설정해둔 데이터에서 (pickle) 레이블과 feature를 뽑아내 각각 설정합니다.
for feature, label in data:
    features.append(feature)
    labels.append(label)

# train과 test set을 설정합니다. test size는 임의로 설정합니다.
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

# SVM 모델을 적용합니다.
model = SVC(C = 1, kernel = 'poly', gamma='auto')
model.fit(xtrain, ytrain)

# 예측값과 정확도를 표현합니다.
prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['Barley', 'etc']

print("정확도 : ", accuracy)
print("예측 카테고리 : ", categories[prediction[0]])

# matplot library를 사용하여 결과물을 출력합니다.
crop_cla = xtest[0].reshape(500, 500)
plt.imshow(crop_cla, cmap='gray')
plt.show()
