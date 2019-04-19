import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(2)
import albumentations as A

class Generator():
    def __init__(self,train,mask,batch_size=32,shuffle=True, aug=False, process=None):
        self.train = train
        self.mask = mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.process = process
        self.agu_method = A.Compose([
                                    A.ElasticTransform(alpha=384, sigma=18, alpha_affine=0),
                                    A.ShiftScaleRotate(rotate_limit=10),
                                    A.HorizontalFlip(),
                                    A.RandomCrop(p=0.5, height=192, width=192),
                                    # A.PadIfNeeded(p=1, min_height=224, min_width=224, border_mode=0)
                                    # A.GridDistortion(),
                                    # A.VerticalFlip(),
                                    # A.RandomRotate90()
                                     ])

        def generator():
            while True:
                if self.shuffle:
                    #打乱数据
                    index = [i for i in range(len(self.mask))]
                    np.random.shuffle(index)
                    X = self.train[index]
                    Y = self.mask[index]

                for i in range(0, len(Y), batch_size):
                    X_batch = X[i:i + self.batch_size]
                    y_batch = Y[i:i + self.batch_size]

                    # X_batch_expand = np.expand_dims(X_batch)
                    if self.aug:
                        for i in range(len(X_batch)):
                            # plt.figure(figsize=(8,8))
                            # plt.subplot(221)
                            # plt.imshow(X_batch[i])
                            # plt.subplot(222)
                            # plt.imshow(np.squeeze(y_batch[i]))
                            agu = self.agu_method(image=np.squeeze(X_batch[i]), mask=np.squeeze(y_batch[i]))
                            # print(X_batch[i].shape, y_batch[i].shape)
                            # (256, 256, 1)(256, 256, 1)
                            # print(agu['image'].shape, agu['mask'].shape)
                            # (256, 256)(256, 256)
                            X_batch[i] = np.expand_dims(agu['image'], axis=-1)
                            y_batch[i] = np.expand_dims(agu['mask'], axis=-1)
                            # X_batch[i] = agu['image']
                            # y_batch[i] = agu['mask']

                            # print(X_batch[i].shape)
                            # plt.subplot(223)
                            # plt.imshow(X_batch[i])
                            # plt.subplot(224)
                            # plt.imshow(np.squeeze(y_batch[i]))
                            # plt.show()
                            # print(i)

                    X_batch = np.array(X_batch)
                    X_batch = self.process(X_batch)
                    y_batch = np.array(y_batch)
                    yield X_batch, y_batch


        self.generator = generator()
        self.steps = len(self.mask) // batch_size
        
        
class Generator_ohem():
    def __init__(self, graph, kerasModel, train, mask, batch_size=32, shuffle=True, aug=False, process=None):
        self.train = train
        self.mask = mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.process = process
        self.agu_method = A.Compose([
                                    A.ElasticTransform(alpha=384, sigma=18, alpha_affine=0),
                                     A.ShiftScaleRotate(rotate_limit=10),
                                     A.HorizontalFlip()
                                     ])
        self.steps = len(self.mask) // batch_size

        self.train_pos = []
        self.mask_pos = []
        self.train_neg = []
        self.mask_neg = []

        for i in range(len(self.mask)):
            if self.mask[i].sum() > 0:
                self.train_pos.append(self.train[i])
                self.mask_pos.append(self.mask[i])
            else:
                self.train_neg.append(self.train[i])
                self.mask_neg.append(self.mask[i])
        self.train_pos = np.array(self.train_pos)
        self.mask_pos = np.array(self.mask_pos)
        self.train_neg = np.array(self.train_neg)
        self.mask_neg = np.array(self.mask_neg)

        self.graph = graph
        self.kerasModel = kerasModel

        # print(len(self.train_pos), len(self.train_neg), '******')

        def generator():
            while True:
                # 打乱数据
                index = [i for i in range(len(self.mask_pos))]
                if self.shuffle:
                    np.random.shuffle(index)
                X_pos = self.train_pos[index]
                Y_pos = self.mask_pos[index]
                # print(self.train_pos.shape, self.mask_pos.shape, '^^^^^^^^^^^^^^^^ori')

                for i in range(0, len(X_pos), int(self.batch_size * 3 / 4)):
                    # print(X_pos.shape, Y_pos.shape, '^^^^^^^^^^^^^^^^ori')
                    X_batch_pos = X_pos[i:i + int(self.batch_size * 3 / 4)]
                    y_batch_pos = Y_pos[i:i + int(self.batch_size * 3 / 4)]
                    # print(X_batch_pos.shape, y_batch_pos.shape, '^^^^^^^^^^^^^^^^pos')

                    X_batch_neg = np.zeros((int(self.batch_size / 4), 192, 192, 1))
                    y_batch_neg = np.zeros((int(self.batch_size / 4), 192, 192, 1))

                    with self.graph.as_default():
                        keras_output = self.kerasModel.predict(self.train_neg)
                        # print(keras_output.shape, '%%%%%%%%%%%%%%%%%')
                    mloss = list()
                    for i in range(len(keras_output)):
                        mloss.append(keras_output[i].sum())

                    topk_indexes = sorted(range(len(mloss)), key=lambda i: mloss[i])[int(-self.batch_size / 4):]
                    # print(topk_indexes)
                    # topk = sorted(mloss)
                    # print(topk)

                    for topk_index in range(int(self.batch_size / 4)):
                        X_batch_neg[topk_index] = self.train_neg[topk_indexes[topk_index]]

                    X_batch = np.concatenate([X_batch_pos, X_batch_neg])
                    y_batch = np.concatenate([y_batch_pos, y_batch_neg])

                    index = [i for i in range(len(X_batch))]
                    np.random.shuffle(index)
                    X_batch = X_batch[index]
                    y_batch = y_batch[index]

                    if self.aug:
                        for i in range(len(X_batch)):
                            agu = self.agu_method(image=np.squeeze(X_batch[i]), mask=np.squeeze(y_batch[i]))
                            X_batch[i] = np.expand_dims(agu['image'], axis=-1)
                            y_batch[i] = np.expand_dims(agu['mask'], axis=-1)

                    X_batch = np.array(X_batch)
                    X_batch = self.process(X_batch)
                    y_batch = np.array(y_batch)
                    # print(X_batch.shape, y_batch.shape, '*************out')
                    yield X_batch, y_batch

        self.generator = generator()
