
# coding: utf-8

# In[ ]:

from tensorflow.python.ops import nn_ops, gen_nn_ops
from tensorflow.python.framework import ops

grad = tf.placeholder("float", [1, 7,7,64])
conv1 = conv2d(x, weights['wc1'], biases['bc1'],'SAME')
pconv1 = activ(conv1)
conv2 = conv2d(pconv1, weights['wc2'], biases['bc2'],'SAME')
pool1 = maxpool(conv2, k=2)
pconv2 = activ(pool1)
conv3 = conv2d(pconv2, weights['wc3'], biases['bc3'],'SAME')
pconv3 = activ(conv3)
conv4 = conv2d(pconv3, weights['wc4'], biases['bc4'],'SAME')
pool2 = maxpool(conv4, k=2)
pconv4 = activ(pool2)
conv5 = conv2d(pconv4, weights['wc5'], biases['bc5'],'SAME')
pconv5 = activ(conv5)
conv6 = conv2d(pconv5, weights['wc6'], biases['bc6'],'VALID')
pool3 = maxpool(conv6, k=2)
pconv6 = activ(pool3)


grad_pre_relu6 = tf.where(0. < grad, gen_nn_ops.relu_grad(grad,pool3), tf.zeros(grad.get_shape()))
grad_conv5=tf.gradients(pool3,pconv5,grad_ys=grad_pre_relu6)[0]

grad_pre_relu5 = tf.where(0. < grad_conv5, gen_nn_ops.relu_grad(grad_conv5,conv5), tf.zeros((1,16,16,64)))
grad_conv4=tf.gradients(conv5,pconv4,grad_ys=grad_pre_relu5)[0]

grad_pre_relu4 = tf.where(0. < grad_conv4, gen_nn_ops.relu_grad(grad_conv4,pool2), tf.zeros((1,16,16,64)))
grad_conv3=tf.gradients(pool2,pconv3,grad_ys=grad_pre_relu4)[0]

grad_pre_relu3 = tf.where(0. < grad_conv3, gen_nn_ops.relu_grad(grad_conv3,conv3), tf.zeros((1,32,32,64)))
grad_conv2=tf.gradients(conv3,pconv2,grad_ys=grad_pre_relu3)[0]

grad_pre_relu2 = tf.where(0. < grad_conv2, gen_nn_ops.relu_grad(grad_conv2,pool1), tf.zeros((1,32,32,32)))
grad_conv1=tf.gradients(pool1, pconv1,grad_ys=grad_pre_relu2)[0]

grad_pre_relu1 = tf.where(0. < grad_conv1, gen_nn_ops.relu_grad(grad_conv1,conv1), tf.zeros((1,64,64,32)))
grad_guided_bp=tf.gradients(conv1,x,grad_ys=grad_pre_relu1)[0]



image = np.reshape(train_x[3],(1,64,64,3))
plt.imshow(image[0])
gradi = sess.run(pconv6,feed_dict = {x:image})
sort = np.argsort(gradi,axis=None)
print(sort.shape)
ind = []
ind.append(np.unravel_index(sort[-1],gradi.shape))
ind.append(np.unravel_index(sort[-111],gradi.shape))
ind.append(np.unravel_index(sort[-211],gradi.shape))
ind.append(np.unravel_index(sort[-181],gradi.shape))
ind.append(np.unravel_index(sort[-311],gradi.shape))
ind.append(np.unravel_index(sort[-1111],gradi.shape))
ind.append(np.unravel_index(sort[-411],gradi.shape))
ind.append(np.unravel_index(sort[-1611],gradi.shape))
ind.append(np.unravel_index(sort[-1411],gradi.shape))
ind.append(np.unravel_index(sort[-151],gradi.shape))

ncols = 6
nrows = 3

fig = plt.figure()
axes = [ fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(1, nrows) for c in range(1, ncols) ]
i=0
for ax in axes:
    naya = np.zeros([1,7,7,64])
    ind1 = ind[i]
    naya[ind1[0]][ind1[1]][ind1[2]][ind1[3]] = gradi[ind1[0]][ind1[1]][ind1[2]][ind1[3]]
    feed_dict_guided_bp={x:image,grad:naya}
    im=sess.run(grad_guided_bp,feed_dict=feed_dict_guided_bp)
    im=(im-im.min())/(im.max()-im.min())
    ax.imshow(im[0])
    i+=1

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

