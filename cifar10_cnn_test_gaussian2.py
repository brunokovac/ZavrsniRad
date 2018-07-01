import tensorflow as tf
import numpy as np
import cifar10_cnn_model as cnn_model

def get_eval_metrics(matrix):
	accuracy = np.trace(matrix) / np.sum(matrix)
	precision = np.diagonal(matrix) / np.sum(matrix, axis=0)
	recall = np.diagonal(matrix) / np.sum(matrix, axis=1)
	
	return accuracy, precision, recall
	
def test(test_data, test_labels):
	logits = cnn_model.model()
	
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, "drive/zavrsni/model")
		
		print("*"*25, "TEST", "*"*25)
		
		softmax_matrix_tf = tf.nn.softmax(logits)
		confusion_matrix_tf = tf.confusion_matrix(labels = Y_, predictions = tf.argmax(logits, 1))
		softmax_matrix, confusion_matrix = sess.run([softmax_matrix_tf, confusion_matrix_tf], feed_dict = {X : test_data, Y_ : test_labels})
		
		np.savetxt("drive/zavrsni/test_labels_gaussian2.txt", test_labels)
		np.savetxt("drive/zavrsni/softmax_matrix_gaussian2.txt", sess.run(logits, feed_dict = {X : test_data, Y_ : test_labels}))
		
		print("Confusion matrix:\n", confusion_matrix)
		
		accuracy, precision, recall = get_eval_metrics(confusion_matrix)
		print("Accuracy:", accuracy)
		print("Precision:", np.average(precision))
		print("Recall:", np.average(recall))
		print("*"*55)
		
import matplotlib.pyplot as plt
def save_image(image, name):
    plt.imshow(image, interpolation="gaussian")
    plt.savefig(name)	

if __name__ == "__main__":
	cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
	cifar_train = (cifar_train[0].astype('float32'), cifar_train[1])
	cifar_test = (cifar_test[0].astype('float32'), cifar_test[1])
		
	train_data = cifar_train[0] / 255
	train_labels = np.asarray(cifar_train[1], dtype=np.int32)
	
	test_data = np.random.normal(scale=0.5, size=cifar_test[0].shape).astype("float32")
	test_labels =  np.zeros(shape=test_data.shape[0], dtype=np.int32)
	
	data_mean = train_data.mean((0, 1, 2))
	data_std = train_data.std((0, 1, 2))

	train_data = (train_data - data_mean) / data_std
	test_data = (test_data - data_mean) / data_std
	
	X = cnn_model.X
	Y_ = cnn_model.Y_
	test(test_data, test_labels)

