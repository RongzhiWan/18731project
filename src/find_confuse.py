import argparse
import sys
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Find confusion matrix given correct output, and predict output')
    parser.add_argument('--predict', dest='predict', type=str, required=True)
    parser.add_argument('--correct', dest='correct', type=str, required=True)
    parser.add_argument('--out', dest='out', type=str, required=True)
    return parser.parse_args()

def main(args):
	args = parse_arguments()
	out_y = np.loadtxt(args.predict, delimiter=',')
	correct_y = np.loadtxt(args.correct, delimiter=',').astype(int)
	num_classes = np.max(correct_y) + 1;

	predict_y = np.argmax(out_y, axis=1)
	confusion = np.zeros([num_classes, num_classes], dtype=int)
	n = correct_y.shape[0]
	correct_num = 0
	for i in range(n):
		predict = predict_y[i]
		correct = correct_y[i]
		confusion[predict, correct] += 1
		if (predict == correct): correct_num += 1
	for i in range(num_classes):
		print('class {} accuracy {}'.format(i, 1.0 * confusion[i, i] / sum(confusion[:, i])))
	print('total accuracy {}'.format(1.0 * correct_num / n))
	np.savetxt(args.out, confusion, fmt='%d', delimiter=',')



if __name__ == '__main__':
    main(sys.argv)