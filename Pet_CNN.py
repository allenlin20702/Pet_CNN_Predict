import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
np.random.seed(10) #隨機產生整數的亂數
from sklearn.metrics import classification_report
    
def show_images_labels_predictions(images,labels,
                                  predictions,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示彩色圖片
        ax.imshow(images[start_id])
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[start_id])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[start_id])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()
    
def show_accuracy_loss(train_history):
    #accuracy準確率
    hist = pd.DataFrame(train_history.history)
    plt.figure(figsize=[8,8])
    plt.subplot(2,1,1)
    plt.plot(hist['accuracy'], 'r' , label='training')
    plt.plot(hist['val_accuracy'], 'b' , label='validate')
    plt.title('Accurary')
    plt.legend()

    #loss損失函數
    plt.figure(figsize=[8,8])
    plt.subplot(2,1,2)
    plt.plot(hist['loss'], 'r' , label='training')
    plt.plot(hist['val_loss'], 'b' , label='validate')
    plt.title('Loss')
    plt.legend()


imagesavepath='Cat_Dog_Dataset/'
try:    
    train_feature=np.load(imagesavepath+'train_feature.npy')  
    test_feature=np.load(imagesavepath+'test_feature.npy')  
    train_label=np.load(imagesavepath+'train_label.npy')      
    test_label=np.load(imagesavepath+'test_label.npy')       
    print("載入 *.npy 檔!") 
    
    # 將 Features 特徵值換為 圖片數量*80*80*3 的 4 維矩陣
    train_feature_vector =train_feature.reshape(len(train_feature), 40,40,3).astype('float32')
    test_feature_vector = test_feature.reshape(len( test_feature), 40,40,3).astype('float32')
    
    #Features 特徵值標準化
    train_feature_normalize = train_feature_vector/255
    test_feature_normalize = test_feature_vector/255
    
    #label 轉換為 One-Hot Encoding 編碼
    train_label_onehot = np_utils.to_categorical(train_label)
    test_label_onehot = np_utils.to_categorical(test_label)
    
    #建立模型
    model = Sequential()
    #建立卷積層1
    model.add(Conv2D(filters=10, 
                      kernel_size=(5,5),
                      padding='same',
                      input_shape=(40,40,3), 
                      activation='relu'))
    
    #建立池化層1
    model.add(MaxPooling2D(pool_size=(2, 2))) #(10,40,40)
    
    # Dropout層防止過度擬合，斷開比例:0.1
    model.add(Dropout(0.2))    
    
    #建立卷積層2
    model.add(Conv2D(filters=10, 
                      kernel_size=(5,5),  
                      padding='same',
                      activation='relu'))
    
    #建立池化層2
    model.add(MaxPooling2D(pool_size=(2, 2))) #(20,20,20)
    
    # Dropout層防止過度擬合，斷開比例:0.2
    model.add(Dropout(0.2))
    
    #建立平坦層：20*20*20=8000 個神經元
    model.add(Flatten()) 
    
    #建立隱藏層
    model.add(Dense(units=512, activation='relu'))
    
    #建立輸出層
    model.add(Dense(units=2,activation='softmax'))
    
    model.summary() #顯示模型
    
    #定義訓練方式
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    """
    https://dotblogs.com.tw/greengem/2017/12/17/094023
    Cross Entropy (categorical_crossentropy)：
    當預測值與實際值愈相近，損失函數就愈小，反之差距很大，
    就會更影響損失函數的值
    Cross Entropy 取代 MSE，
    因為在梯度下時，Cross Entropy 計算速度較快，
    其他變形包括 sparse_categorical_crossentropy、binary_crossentropy。
    """
    
    #以(train_feature_normalize,train_label_onehot)資料訓練，
    #訓練資料保留 20%作驗證,訓練10次、每批次讀取200筆資料，顯示簡易訓練過程
    train_history =model.fit(x=train_feature_normalize,
                             y=train_label_onehot,validation_split=0.2, 
                             epochs=10, batch_size=200,verbose=2)
    #評估準確率
    scores = model.evaluate(test_feature_normalize, test_label_onehot)
    print('\n準確率=',scores[1])
        
    #預測
    prediction = np.argmax(model.predict(test_feature_normalize), axis=-1)
    #print(prediction)
    
    del model    
    
    #顯示圖像、預測值、真實值 
    show_images_labels_predictions(test_feature,test_label,prediction,0)
    
    #accuracy準確率 loss損失函數
    show_accuracy_loss(train_history)
    
    #confusion matrix
    print("\n",pd.crosstab(test_label,prediction,rownames=['actual label'],colnames=['prediction']),"\n")

    #顯示precision, recall, f1-score
    print('classification')
    print(classification_report(test_label,prediction))
    
except:
    print(".npy 檔未建立!")    
