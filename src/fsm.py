import argparse
import sys
import numpy as np
import glob

class FSM():
    """docstring for ClassName"""
    def __init__(self):
        # state list that stores each states. First state indicates the starting state
        self.states = [np.array([])]
        # list of set that stores transition from each state to another
        self.trans = [set()]

    def _find_state_num(self, state):
        for i in range(len(self.states)):
            if np.all(self.states[i] == state):
                return i
        return None

    # return new_state's number
    def _add_transition(self, old_state_num, new_state):
        # find new state in state list
        new_state_num = self._find_state_num(new_state)
        if (new_state_num == None):
            new_state_num = len(self.states)
            self.states.append(new_state)
            self.trans.append(set())
        self.trans[old_state_num].add(new_state_num)
        return new_state_num

    def _find_transition(self, old_state_num, new_state):
        # find new state in state list
        new_state_num = self._find_state_num(new_state)
        # print("total states", len(self.states), "new state num", new_state_num)
        if (new_state_num == None):
            return (False, 0)
        return (new_state_num in self.trans[old_state_num], new_state_num)

    def add_translist(self, transitions):
        new_state_num = 0
        for i in range(transitions.shape[0]):
            old_state_num = new_state_num
            new_state_num = self._add_transition(old_state_num, transitions[i, :])
        # print("transition num", transitions.shape[0], "state num", len(self.states))

    # return percentage of transitions that can be found
    def find_translist_innum(self, transitions):
        total_trans_num = transitions.shape[0]
        in_trans_num = 0
        new_state_num = 0
        for i in range(total_trans_num):
            old_state_num = new_state_num
            (can_find, new_state_num) = self._find_transition(old_state_num, transitions[i, :])
            if (can_find): 
                in_trans_num += 1
        return 1.0 * in_trans_num / total_trans_num

    def save(self, file_prefix):
        np.save(file_prefix + '_states.npy', self.states)
        np.save(file_prefix + '_trans.npy', self.trans)

    def load(self, file_prefix):
        self.states = np.load(file_prefix + '_states.npy')
        self.trans = np.load(file_prefix + '_trans.npy')

class FSMLearn():
    def __init__(self, num_classes, save_prefix=None):
        self.num_classes = num_classes
        self.save_prefix = save_prefix
        
        self.fsm = [None] * num_classes
        for i in range(num_classes):
            self.fsm[i] = FSM()

        if (self.save_prefix != None):
            for i in range(self.num_classes):
                self.fsm[i].load('{}_class{}'.format(self.save_prefix, i))


    def train(self, train_X, train_Y):
        for i in range(len(train_Y)):
            data_x = train_X[i]
            data_y = train_Y[i]
            self.fsm[data_y].add_translist(data_x)
        if (self.save_prefix != None):
            for i in range(self.num_classes):
                self.fsm[i].save('{}_class{}'.format(self.save_prefix, i))

    def test(self, test_X):
        out_Y = np.zeros([len(test_X), self.num_classes])
        for i in range(len(test_X)):
            data_x = test_X[i]
            for j in range(self.num_classes):
                out_Y[i, j] = self.fsm[j].find_translist_innum(data_x)
        return out_Y

def parse_arguments():
    parser = argparse.ArgumentParser(description='Device-type identification from packet features using FSM')
    parser.add_argument('--data_folder', dest='data_folder', type=str, default='../data/output/v1')
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    X = []
    Y = []
    glob_find = '{}/*.txt'.format(args.data_folder)
    c = 0
    for file in glob.glob(glob_find):
        if file.find("README") != -1:
            continue
        with open(file) as f:
            for row in f.readlines():
                c += 1
                data = row.split('\t')
                
                num_classes = int(data[0])
                Y.append(num_classes)

                row_num = int(data[1])
                col_num = int(data[2])
                flatten_matrix = np.fromstring(data[3], dtype=int, sep=' ')
                mat = np.array(flatten_matrix).reshape((row_num, col_num))
                mat = mat.astype(np.float)
                X.append(mat)
    num_classes = np.amax(Y) + 1
    fsm_learn = FSMLearn(num_classes)
    # just a bit testing
    train_X = X[2:20]
    train_Y = Y[2:20]
    print("train Y", train_Y)
    fsm_learn.train(train_X, train_Y)
    test_X = X[0:2]
    print("test Y", Y[0:2])
    test_Y_out = fsm_learn.test(test_X)
    print("test Y out", test_Y_out)



if __name__ == '__main__':
    main(sys.argv)