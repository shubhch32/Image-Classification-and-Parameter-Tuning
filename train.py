import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.python.ops import nn_ops, gen_nn_ops
from tensorflow.python.framework import ops
import argparse


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        '--lr',type=float, required=True)
    parser.add_argument(
        '--batch_size',type=int,required=True)
    parser.add_argument(
        '--init',type=int,required=True)
    parser.add_argument(
        '--save_dir',type=str,required=True)
    parser.add_argument(
        '--epochs',type=int,required=True)
    parser.add_argument(
        '--dataAugment',type=int,required=True)
    parser.add_argument(
        '--train',type=str,required=True)
    parser.add_argument(
        '--val',type=str,required=True)
    parser.add_argument(
        '--test',type=str,required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    lr = args.lr
    batch_size=args.batch_size
    init=args.init
    save_dir=args.save_dir
    epochs=args.epochs
    dataAugment=args.dataAugment
    train=args.train
    val=args.val
    test=args.test
    # Return all variable values
    return lr,batch_size,init,save_dir,epochs,dataAugment,train,val,test


lr,batch_size,init,save_dir,epochs,dataAugment,train,val,test = get_args()

train=pd.read_csv(train)
test=pd.read_csv(test)
valid=pd.read_csv(val)
train = train.drop('id',axis=1)
valid=valid.drop('id',axis=1)
test = test.drop('id',axis=1)

np.random.seed(1234)
x_train = np.asarray(train.loc[:, train.columns != 'label'])
divi = np.max(x_train,axis=0)-np.min(x_train,axis=0)+1e-8
means = np.mean(x_train,axis=0)
# x_train=x_train-means
vari = np.sum(x_train**2,axis=0)/x_train.shape[0]
vari = np.sqrt(vari)
x_train = x_train/vari
train_x = np.reshape(x_train,(-1,64,64,3))
train_y = np.asarray(train['label'])
# train_x = train_x[0:10]
# train_y = train_y[0:10]
# train_y = np.reshape(train_y,(train_y.shape[0],1))

x_valid = np.asarray(valid.loc[:, valid.columns != 'label'])
# x_valid = x_valid-means
x_valid = x_valid/vari
valid_x = np.reshape(x_valid,(-1,64,64,3))
y_valid1 = np.asarray(valid['label'])
valid_y=[]
for i in range(x_valid.shape[0]):
    valid_y.append(np.eye(20)[y_valid1[i]])
valid_y=np.asarray(valid_y)

x_test = np.asarray(test.loc[:, valid.columns != 'label'])
# x_test = x_test-means
x_test = x_test/vari
test_x = np.reshape(x_test,(-1,64,64,3))

np.random.seed(1234)
tf.random.set_random_seed(1234)
training_iters = epochs
learning_rate = lr
n_fc1=4096
n_fc2=1024
n_input = 64
n_classes = 20

x = tf.placeholder("float", [None, 64,64,3])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, pad):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x,alpha=0.1) 

def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def flip(train_x):
    result = []
    X = tf.placeholder(tf.float64, shape=(None, 64 ,64,3))
    train_x_flip = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        result = sess.run(train_x_flip, feed_dict={X: train_x})
    return result

def flip1(train_x):
    result = []
    X = tf.placeholder(tf.float64, shape=(None, 64 ,64,3))
    train_x_flip1 = tf.image.flip_up_down(X)
    with tf.Session() as sess:
        result = sess.run(train_x_flip1, feed_dict={X: train_x})
    return result
  
def scale1(train_x):
    result = []
    original_size = [64, 64]
    X = tf.placeholder(dtype = tf.float64, shape = (None, 64 ,64,3))
    #crop_size = [8769, 22, 22, 3]
    seed = np.random.randint(1234)
    X1 = tf.random_crop(X, size = (train_x.shape[0],32,32,3), seed = seed)
    #X1=tf.image.central_crop(X, 0.5)
    train_x_scale1 = tf.image.resize_images(X1, size = original_size)
    with tf.Session() as sess:
        result = sess.run(train_x_scale1, feed_dict={X: train_x})
    return result

def scale2(train_x):
    result = []
    original_size = [64, 64]
    X = tf.placeholder(dtype = tf.float64, shape = (None, 64 ,64,3))
    #crop_size = [8769, 44, 44, 3]
    seed = np.random.randint(1234)
    X1 = tf.random_crop(X, size = (train_x.shape[0],48,48,3), seed = seed)
    #X1=tf.image.central_crop(X, 0.75)
    train_x_scale2 = tf.image.resize_images(X1, size = original_size)
    with tf.Session() as sess:
        result = sess.run(train_x_scale2, feed_dict={X: train_x})
    return result

def rotate(train_x):
    result = []
    X = tf.placeholder(tf.float64, shape=(None, 64 ,64,3))
    train_x_rotate = tf.image.rot90(X, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    with tf.Session() as sess:
        result = sess.run(train_x_rotate, feed_dict={X: train_x})
    return result

# def trans(train_x):
    

def bat_norm(lay):
    return tf.layers.batch_normalization(lay, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(), training=True, trainable=True, renorm=False, renorm_momentum=0.99)

def conv_net(x, weights, biases):  
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],'SAME')
    bat1 = bat_norm(conv1)
    conv2 = conv2d(bat1, weights['wc2'], biases['bc2'],'SAME')
    bat2= bat_norm(conv2)
    pool1 = maxpool(bat2, k=2)
    conv3 = conv2d(pool1, weights['wc3'], biases['bc3'],'SAME')
    bat3= bat_norm(conv3)
    conv4 = conv2d(bat3, weights['wc4'], biases['bc4'],'VALID')
    bat4= bat_norm(conv4)
    pool2 = maxpool(bat4, k=2)
    
    conv5 = conv2d(pool2, weights['wc5'], biases['bc5'],'SAME')
    bat5= bat_norm(conv5)
    conv6 = conv2d(bat5, weights['wc6'], biases['bc6'],'VALID')
    bat6= bat_norm(conv6)
    conv6 = maxpool(bat6, k=2)
    
    fc = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
#     bn1 = tf.nn.batch_normalization(fc1,0.0,1.0,gamma['fcg1'],beta['fcb1'],1e-8)
#     bn1 = bat_norm(fc1)
#     drop1 = tf.nn.dropout(bn1,keep_prob=0.5)
#     bn1 = tf.nn.batch_normalization(drop1,tf.nn.moments(drop1, axes=[0])[0],tf.nn.moments(drop1, axes=[0])[1],gamma['fcg1'],beta['fcb1'],1e-8)
#     bn1 = tf.nn.batch_normalization(drop1,0.0,1.0,gamma['fcg1'],beta['fcb1'],1e-8)
#     bn1 = tf.layers.batch_normalization(drop1,training = True)
    fc1 = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fcl1 = tf.nn.leaky_relu(fc1,alpha=0.1)
    
#     bn2 = tf.nn.batch_normalization(fc2,0.0,1.0,gamma['fcg2'],beta['fcb2'],1e-8)
    bn2 = bat_norm(fcl1)
    drop2 = tf.nn.dropout(bn2,keep_prob=0.8)
#     bn2 = tf.nn.batch_normalization(drop2,tf.nn.moments(drop2, axes=[0])[0],tf.nn.moments(drop2, axes=[0])[1],gamma['fcg2'],beta['fcb2'],1e-8)
#     bn2 = tf.nn.batch_normalization(drop2,0.0,1.0,gamma['fcg2'],beta['fcb2'],1e-8)
#     bn2 = tf.layers.batch_normalization(drop2,training = True)
    fc3 = tf.add(tf.matmul(drop2, weights['wd2']), biases['bd2'])
    fcl3 = tf.nn.leaky_relu(fc3,alpha=0.1)
    bn3 = bat_norm(fcl3)
    drop3 = tf.nn.dropout(bn3,keep_prob=0.8)
    out = tf.add(tf.matmul(drop3, weights['out']), biases['out'])
    return x,fc,out


if init==2:
    weights = {
        'wc1': tf.get_variable('W0', shape=(7,7,3,32), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)), 
        'wc2': tf.get_variable('W1', shape=(7,7,32,64), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)), 
        'wc3': tf.get_variable('W2', shape=(5,5,64,128), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)), 
        'wc4': tf.get_variable('W3', shape=(5,5,128,256), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'wc5': tf.get_variable('W4', shape=(3,3,256,512), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'wc6': tf.get_variable('W5', shape=(3,3,512,512), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)), 
        'wd1': tf.get_variable('W6', shape=(18324,n_fc1), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'wd2': tf.get_variable('W7', shape=(n_fc1,n_fc2), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'out': tf.get_variable('W8', shape=(n_fc2,n_classes), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)), 
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bc4': tf.get_variable('B3', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bc5': tf.get_variable('B4', shape=(512), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bc6': tf.get_variable('B5', shape=(512), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bd1': tf.get_variable('B6', shape=(n_fc1), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'bd2': tf.get_variable('B7', shape=(n_fc2), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'out': tf.get_variable('B8', shape=(20), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
    }
else:
    weights = {
        'wc1': tf.get_variable('W0', shape=(7,7,3,32), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)), 
        'wc2': tf.get_variable('W1', shape=(7,7,32,64), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)), 
        'wc3': tf.get_variable('W2', shape=(5,5,64,128), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)), 
        'wc4': tf.get_variable('W3', shape=(5,5,128,256), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'wc5': tf.get_variable('W4', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'wc6': tf.get_variable('W5', shape=(3,3,512,512), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)), 
        'wd1': tf.get_variable('W6', shape=(18324,n_fc1), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'wd2': tf.get_variable('W7', shape=(n_fc1,n_fc2), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'out': tf.get_variable('W8', shape=(n_fc2,n_classes), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)), 
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bc4': tf.get_variable('B3', shape=(256), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bc5': tf.get_variable('B4', shape=(512), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bc6': tf.get_variable('B5', shape=(512), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bd1': tf.get_variable('B6', shape=(n_fc1), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'bd2': tf.get_variable('B7', shape=(n_fc2), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
        'out': tf.get_variable('B8', shape=(20), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.dtypes.float32)),
    }

acti1, acti, pred = conv_net(x, weights, biases)
test_pred = tf.argmax(pred,1)
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
lamb=0.01
regularizer = tf.add_n([tf.nn.l2_loss(weights['wd1']),tf.nn.l2_loss(weights['wd2']),tf.nn.l2_loss(weights['out']),tf.nn.l2_loss(biases['bd1']),tf.nn.l2_loss(biases['bd2']),tf.nn.l2_loss(biases['out'])])
# regularizer = tf.nn.l2_loss(weights['wd1'])+tf.nn.l2_loss(weights['wd2'])+tf.nn.l2_loss(weights['out'])
cost = tf.add_n([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) , lamb * regularizer])
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
initialize = tf.global_variables_initializer()
max_acc = 0.0

print("Done 0")
if(dataAugment==1):
    train_xf = np.concatenate([train_x,flip1(train_x),rotate(train_x)], axis = 0)
else:
    train_xf = np.asarray(train_x)
print("Done 1")
if(dataAugment==1):
    train_y1f = np.concatenate([train_y,train_y,train_y], axis = 0)
else:
    train_y1f = np.asarray(train_y)
print("Done 2")

train_xf,train_y1f = shuffle(train_xf,train_y1f,random_state=0)

train_yf=[]
for i in range(train_xf.shape[0]):
    train_yf.append(np.eye(20)[train_y1f[i]])
train_yf=np.asarray(train_yf)

print("Done 3")
if(dataAugment==1):
    train_xs = np.concatenate([flip(train_x), scale2(train_x)], axis = 0)
else:
    train_xs = np.empty(shape=(0))
print(np.asarray(train_xs.shape))
print("Done 4")
if(dataAugment==1):
    train_y1s = np.concatenate([train_y, train_y], axis = 0)
else:
    train_y1s = np.empty(shape=(0))
print(np.asarray(train_y1s.shape))
print("Done 5")
if(dataAugment==1):
    train_xs,train_y1s = shuffle(train_xs,train_y1s,random_state=0)

train_ys=[]
for i in range(train_xs.shape[0]):
    train_ys.append(np.eye(20)[train_y1s[i]])
train_ys=np.asarray(train_ys)
saver = tf.train.Saver(max_to_keep=training_iters)
with tf.Session() as sess:
    sess.run(initialize) 
    train_loss = []
    valid_loss = []
    train_accuracy = []
    test_accuracy = []
    train_acc = 0.0
    min_val_loss=1000
    p=0
    for i in range(training_iters):
        t_loss,t_acc = [],[]
        for batch in range(len(train_xf)//batch_size):
            batch_x = train_xf[batch*batch_size:min((batch+1)*batch_size,len(train_xf))]
            batch_y = train_yf[batch*batch_size:min((batch+1)*batch_size,len(train_yf))]
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        for batch in range(len(train_xs)//batch_size):
            batch_x = train_xs[batch*batch_size:min((batch+1)*batch_size,len(train_xs))]
            batch_y = train_ys[batch*batch_size:min((batch+1)*batch_size,len(train_ys))]
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
        for batch in range(len(train_xf)//batch_size):
            batch_x = train_xf[batch*batch_size:min((batch+1)*batch_size,len(train_xf))]
            batch_y = train_yf[batch*batch_size:min((batch+1)*batch_size,len(train_yf))]
            loss, acc =  sess.run([cost1, accuracy], feed_dict={x: batch_x, y: batch_y})
            t_loss.append(loss)
            t_acc.append(acc)
        
        for batch in range(len(train_xs)//batch_size):
            batch_x = train_xs[batch*batch_size:min((batch+1)*batch_size,len(train_xs))]
            batch_y = train_ys[batch*batch_size:min((batch+1)*batch_size,len(train_ys))]
            loss, acc =  sess.run([cost1, accuracy], feed_dict={x: batch_x, y: batch_y})
            t_loss.append(loss)
            t_acc.append(acc)
        
        loss=sum(t_loss)/len(t_loss)
        acc=sum(t_acc)/len(t_acc)
        print("Iter " + str(i) + ", Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc))
        print("Optimization scale Finished!")
        valid_acc,val_loss = sess.run([accuracy,cost1], feed_dict={x: valid_x,y : valid_y})
        train_loss.append(loss)
        valid_loss.append(val_loss)
        train_accuracy.append(acc)
        test_accuracy.append(valid_acc)
        print("Iter " + str(i) + ", Valid Loss= " + "{:.4f}".format(val_loss) + ", Valid Accuracy= " + \
                      "{:.4f}".format(valid_acc))
#         print("Validation scale Accuracy:","{:.4f}".format(valid_acc))
        if(min_val_loss>val_loss):
            y_pred = sess.run(test_pred,feed_dict={x: test_x})
            p=0
        else:
            p+=1
            if(p>=5):
                break
        # y_pred_test = np.zeros([y_pred.shape[0],2],dtype=int)
        # for j in range (y_pred.shape[0]):
        #     y_pred_test[j][0]=j
        #     y_pred_test[j][1]=y_pred[j]
        # df = pd.DataFrame(y_pred_test,columns=['id','label'])
        # df.to_csv('df'+str(i)+'.csv',index=False)
        save_path = saver.save(sess,save_dir+"models/epoch",global_step = i)
#         print(save_path)
    filters = sess.run(weights['wc1'])