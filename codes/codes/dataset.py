
# ---------------
import os
import numpy as np
# import scipy.misc as misc
import scipy.io as sio

import torch
import torch.utils.data as data

# dataset augmentation
# mirror, offset, one-pixel translation
def augment(l):
    # mode = np.random.randint(0, 4)
    mode = np.random.randint(0, 3)
    # print(mode)
    def _augment(x, mode=0):
        if mode == 0:
            return x
        elif mode == 1:
            return -1*x
        elif mode == 2:
            return np.flip(x)
        # elif mode == 3:
        #     return x+np.random.uniform(0,0.5,1)
    return [_augment(_l, mode=mode) for _l in l]


def onehot2scalar(y):
    '''
        Docs: transform one-hot label to 0/1, [1,0]==>0, [0,1]==>1
        Args: y is an array, with form as (n, one-hot-label)
    '''
    y_bin = np.argmax(y, axis=1)
    return y_bin.astype(np.uint8)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

class CancerDetection(data.Dataset):
    def __init__(self, args, mode='train', isOnehot=False):
        self.mode = mode
        self.isOnehot = isOnehot
        self.isCNN = args.isCNN
        assert args.task in ['Task_1', 'Task_2']
        self.task = args.task
        self.data = sio.loadmat(os.path.join(args.dir_root, args.task, args.filename))

        if self.mode == 'train':
            self.repeat = 1
            # x with shape as (n_data, n_dim); y with shape as (n_data, one_hot)
            self.x = self.data['X_tr']
            self.y = self.data['Y_tr']
            if self.isOnehot:
                self.y = one_hot(self.y,2)

        elif self.mode == 'test':
            self.x = self.data['X_test']
            self.y = self.data['Y_test']
            if self.isOnehot:
                self.y = one_hot(self.y,2)

        elif self.mode == 'validation':
            self.x = self.data['X_val']
            self.y = self.data['Y_val']
            if self.isOnehot:
                self.y = one_hot(self.y,2)

        elif self.mode == 'unique':
            self.x = self.data['X_test_uni']
            self.y = self.data['Y_test_uni']
            if self.isOnehot:
                self.y = one_hot(self.y,2)
        else:
            print('Unvalid Dataset Mode, Pls select from [train, test, validation, unique] !')
            assert False
        
    def __len__(self):
        return self.x.shape[0]

    def _get_index(self, idx):
        return idx

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        spctr = self.x[idx]
        label = self.y[idx]
        assert spctr.shape == (779,)
        # preporcessing
        if self.mode == 'train':
            if self.task == 'Task_2':
                if label[0] == 0:
                    spctr = augment([spctr])[0].copy()
                else:
                    if np.random.randint(0, 2) == 0:
                        spctr = augment([spctr])[0].copy()
            else:
                # if label[0] == 0:
                spctr = augment([spctr])[0].copy()

        if self.isCNN:
            return spctr.reshape(-1,779), label
        else:
            return spctr, label

    # ## visualization
    # def spctr_visualization(self, spctr, label):
    #     if self.isOnehot == True:
    #         label = np.matmul(label,np.array([[0],[1]]))

    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1,1,1)
    #     ax.plot(spctr,'r' if label==1 else 'b',label=label)
    #     ax.legend()
    #     plt.show()


# class CancerDetection_CNN(data.Dataset):
#     def __init__(self, args, mode='train', isOnehot=False):
#         self.args = args
#         self.mode = mode
#         self.isOnehot = isOnehot
#         self.data = sio.loadmat(os.path.join(args.dir_root, args.filename))

#         if self.mode == 'train':
#             # x with shape as (n_data, n_dim); y with shape as (n_data, one_hot)
#             self.x = self.data['X_tr']
#             self.y = self.data['Y_tr']
#             if self.isOnehot:
#                 self.y = one_hot(self.y,2)

#         elif self.mode == 'test':
#             self.x = self.data['X_test']
#             self.y = self.data['Y_test']
#             if self.isOnehot:
#                 self.y = one_hot(self.y,2)

#         elif self.mode == 'validation':
#             self.x = self.data['X_val']
#             self.y = self.data['Y_val']
#             if self.isOnehot:
#                 self.y = one_hot(self.y,2)

#         elif self.mode == 'unique':
#             self.x = self.data['X_test_uni']
#             self.y = self.data['Y_test_uni']
#             if self.isOnehot:
#                 self.y = one_hot(self.y,2)
#         else:
#             print('Unvalid Dataset Mode, Pls select from [train, test, validation, unique] !')
#             assert False
        
#     def __len__(self):
#         return self.x.shape[0]

#     def _get_index(self, idx):
#         return idx

#     def __getitem__(self, idx):
#         idx = self._get_index(idx)
#         spctr = self.x[idx]
#         label = self.y[idx]
#         assert spctr.shape == (779,)
#         # preporcessing
#         if self.mode == 'train':
#             if label[0] == 0:
#                 spctr = augment([spctr])[0].copy()
#             else:
#                 if np.random.randint(0, 2) == 0:
#                     spctr = augment([spctr])[0].copy()
#         return spctr.reshape(-1,779), label


if __name__=='__main__':
    # Prepare Data
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir_root', default='./data')
    parser.add_argument('--filename', default='dataset_01_02.mat')
    args = parser.parse_args()

    dataset = CancerDetection(args, mode='train', isOnehot=False)

    spctr, label = dataset.__getitem__(0)
    import matplotlib.pyplot as plt; import pdb; pdb.set_trace()

    pass


