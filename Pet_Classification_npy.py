import os,cv2,glob
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(10)
    
# 將原始圖片 resize 後存在 images 串列，標籤存在 labels 串列
images=[]
labels=[]
dict_labels = {"Cat":0, "Dog":1}
size = (40,40) #由於原始資料影像大小不一，因此制定一個統一值
for folders in glob.glob("kagglecatsanddogs_5340/PetImages/*"):
    print(folders,"圖片讀取中…") 
    # 只讀取貓、狗圖片
    for filename in os.listdir(folders):
        label=folders.split("/")[-1] 
        #原先label=folders.split("//")[-1]： "//" 要改成 "/"，這樣才能完整切割檔名 Cat Dog
        #print(label) 檢查有無完整切割檔名 Cat Dog
        try:
            img=cv2.imread(os.path.join(folders,filename))
            img = cv2.resize(img,dsize=size)
            if img is not None:
                images.append(img)
                labels.append(dict_labels[label])
                #print(labels)
        except:
            print(os.path.join(folders,filename),"無法讀取!")
            pass


print(len(images),len(labels))  # 24946 24946 


#建立訓練資料和測試資料，包括訓練特徵集、測試特徵集、訓練標籤和測試標籤	        
train_feature,test_feature,train_label,test_label = \
    train_test_split(images,labels,test_size=0.2,random_state=42) 
    
train_feature=np.array(train_feature) # 串列轉為矩陣 
test_feature=np.array(test_feature)   # 串列轉為矩陣 
train_label=np.array(train_label)     # 串列轉為矩陣   
test_label=np.array(test_label)       # 串列轉為矩陣 

print(len(train_feature),len(test_feature))  # 19956 4990
print(train_feature.shape,train_label.shape) # (19956, 40, 40, 3) (19956,)
print(test_feature.shape,test_label.shape)   # (4990, 40, 40, 3) (4990,)

imagesavepath='Cat_Dog_Dataset/'
if not os.path.exists(imagesavepath):
    os.makedirs(imagesavepath)
np.save(imagesavepath+'train_feature.npy',train_feature)    
np.save(imagesavepath+'test_feature.npy',test_feature)  
np.save(imagesavepath+'train_label.npy',train_label)    
np.save(imagesavepath+'test_label.npy',test_label)  

print('train_feature.npy 已儲存')
print('test_featurel.npy 已儲存')
print('train_label.npy 已儲存')
print('test_label.npy 已儲存')