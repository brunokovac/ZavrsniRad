import numpy as np
import matplotlib.pyplot as plt

def get_correct_and_wrong_means(data_type, softmax_values, labels):
	total = softmax_values.shape[0]
	print(data_type, "pred. prob (mean):", np.mean(np.max(softmax_values, 1)))
	print("Accuracy ({}):".format(data_type.lower()), np.sum(np.argmax(softmax_values, 1) == labels) / total)
	
def draw_ROC(data_type, P, N):
	assert(P.shape == N.shape)

	total = P.shape[0]
	P = [[np.max(x), True] for x in P]
	N = [[np.max(x), False] for x in N]
	
	union = np.append(P, N, axis=0)
	union = sorted(union, key=lambda x : -x[0])

	t1 =[]; t2 = []
	for i in range(1, len(union)+1):
		used = union[:i]
		
		tp = sum(x[1] for x in used)
		fp = len(used) - tp
		
		t1.append(fp / total)
		t2.append(tp / total)
		
	area = 0
	for i in range(len(t1) - 1):
		area += np.trapz(np.array(t2[i:i+2]), dx=abs(t1[i] - t1[i+1]))
		
	plt.title("ROC krivulja {0}\nAUROC = {1:.2f}%".format(data_type, area * 100))
	plt.xlabel("Udio lažnih pozitiva (false positive rate)")
	plt.ylabel("Udio točnih pozitiva (true positive rate)")
	plt.plot(t1, t2, 'r-', linewidth=2.5)
	plt.show()
	

softmax_original = np.loadtxt("softmax_matrix_original.txt")
softmax_gaussian = np.loadtxt("softmax_matrix_gaussian2.txt")
softmax_flipped = np.loadtxt("softmax_matrix_flipped.txt")

get_correct_and_wrong_means("Original", softmax_original, np.loadtxt("test_labels_original.txt"))
print()
get_correct_and_wrong_means("Gaussian", softmax_gaussian, np.loadtxt("test_labels_gaussian2.txt"))
print()
get_correct_and_wrong_means("Flipped", softmax_flipped, np.loadtxt("test_labels_flipped.txt"))

draw_ROC("Gaussovog šuma", softmax_original, softmax_gaussian)
#draw_ROC("obrnutih slika", softmax_original, softmax_flipped)
