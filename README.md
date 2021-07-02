# Africal wildlife Prediction using CNN

Pretrained VGG Prediction over African Wildlife 

![](https://github.com/sagnik1511/African-Wildlife-Animal-Recognition-with-CNN/blob/main/assets/heading.jpg)
---

### Dataset :
  
   Link : [https://www.kaggle.com/biancaferreira/african-wildlife](https://www.kaggle.com/biancaferreira/african-wildlife)
   
   Data directory :
   
                              root/
                              ├─ buffalo/
                                 ├─ 001.jpg
                                 ├─ 001.txt
                              elephant/
                                 ├─ 001.jpg
                                 ├─ 001.txt
                              rhino/
                                 ├─ 001.jpg
                                 ├─ 001.txt
                              zebra/
                                 ├─ 001.jpg
                                 ├─ 001.txt


### Model Evaluation :

![](https://github.com/sagnik1511/African-Wildlife-Animal-Recognition-with-CNN/blob/main/assets/model%20metrics.png)

### Prediction :

**Train accuracy is near : 99.2%**

**Validation accuracy is near : 86%**

#### Predict function :

                                     def predict(path):
                                          img = cv2.imread( path )
                                          img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
                                          img = cv2.resize( img , (256,256) )
                                          img = img.reshape(1 , 256 , 256 , 3)
                                          pred = np.argmax( clf.predict(img) )
                                          plt.imshow(img.reshape( 256 , 256 ,3))
                                          plt.title(f'Prediction : {classes[pred]}')
                                          
                                          
 #### Results :
 
 ![](https://github.com/sagnik1511/African-Wildlife-Animal-Recognition-with-CNN/blob/main/assets/pred1.png)
 ![](https://github.com/sagnik1511/African-Wildlife-Animal-Recognition-with-CNN/blob/main/assets/pred2.png)
 
 
 
 # Do Star the repository :)
 # Thank Yoy for visiting :)
