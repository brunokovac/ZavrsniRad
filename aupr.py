import numpy as np
import matplotlib.pyplot as plt
	
def draw_PR_curve(data_type, in_out, P, N):
	assert(P.shape == N.shape)

	P = [[np.max(x) if in_out=="in" else -np.max(x), True] for x in P]
	N = [[np.max(x) if in_out=="in" else -np.max(x), False] for x in N]
	
	union = np.append(P, N, axis=0)
	union = sorted(union, key=lambda x : -x[0])

	t1 =[]; t2 = []
	for i in range(1, len(union)):
		used = union[:i]
		
		tp = sum(x[1] for x in used)
		fp = len(used) - tp
		fn = total - tp
		
		if (tp + fp) != 0:
			t1.append(tp / (tp + fn))
			t2.append(tp / (tp + fp))
	
	area = 0
	for i in range(len(t1) - 1):
		area += np.trapz(np.array(t2[i:i+2]), dx=abs(t1[i] - t1[i+1]))
		
	plt.title("PR krivulja {0}\nAUPR {1} = {2:.2f}%".format(data_type, in_out, area * 100))
	plt.xlabel("Odziv (recall)")
	plt.ylabel("Preciznost (precision)")
	axes = plt.gca()
	axes.set_xlim([0, 1.05])
	axes.set_ylim([0, 1.05])
	plt.plot(t1, t2, 'r-', linewidth=2.5)
	plt.show()
	

softmax_original = np.loadtxt("softmax_matrix_original.txt")
softmax_gaussian = np.loadtxt("softmax_matrix_gaussian2.txt")
softmax_flipped = np.loadtxt("softmax_matrix_flipped.txt")
labels = np.loadtxt("test_labels_original.txt")

total = len(labels)

#draw_PR_curve("Gaussovog šuma", "in", softmax_original, softmax_gaussian)
#draw_PR_curve("obrnutih slika", "in", softmax_original, softmax_flipped)

draw_PR_curve("Gaussovog šuma", "out", softmax_gaussian, softmax_original)
#draw_PR_curve("obrnutih slika", "out", softmax_flipped, softmax_original)
