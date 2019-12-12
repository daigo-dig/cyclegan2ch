import scipy
from glob import glob
import numpy as np
import cv2
from operator import itemgetter

class DataLoader():
    def __init__(self, dataset_name,img_rows=256,img_cols=256,img_channelsA=2,img_channelsB=1):
        self.dataset_name = dataset_name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channelsA = img_channelsA
        self.img_channelsB = img_channelsB
        self.img_res = (img_rows,img_cols)
        self.img_shapeA = (img_rows,img_cols,img_channelsA)
        self.img_shapeB = (img_rows,img_cols,img_channelsB)

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = np.reshape(cv2.resize(img, self.img_res),self.img_shapeB)
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = np.reshape(cv2.resize(img, self.img_res),self.img_shapeB)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_dataA(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path1 = glob('./datasets/%s/%s%s/*' % (self.dataset_name, data_type, "1"))
        path2 = glob('./datasets/%s/%s%s/*' % (self.dataset_name, data_type, "2"))
        data_size = min(len(path1), len(path2))
        shuffled_idx = np.random.choice(np.arange(data_size), batch_size, replace=False)

        imgs = []
        for img_path in shuffled_idx:
            img1 = self.imread(path1[img_path])
            img2 = self.imread(path2[img_path])
            img = np.stack([img1,img2])
            if not is_testing:
                img = np.reshape(img,self.img_shapeA)
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = np.reshape(img,self.img_shapeA)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A1 = glob('./datasets/%s/%sA1/*' % (self.dataset_name, data_type))
        path_A2 = glob('./datasets/%s/%sA2/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A1), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        data_sizeA = len(path_A1)
        data_sizeB = len(path_B)
        shuffled_idxA = np.random.choice(np.arange(data_sizeA), self.n_batches, replace=False)
        shuffled_idxB = np.random.choice(np.arange(data_sizeB), self.n_batches, replace=False)
        path_A1 = itemgetter(*shuffled_idxA)(path_A1)
        path_A2 = itemgetter(*shuffled_idxA)(path_A2)
        path_B = itemgetter(*shuffled_idxB)(path_B)

        for i in range(self.n_batches-1):
            batch_A1 = path_A1[i*batch_size:(i+1)*batch_size]
            batch_A2 = path_A2[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A1, img_A2, img_B in zip(batch_A1, batch_A2, batch_B):
                img_A1 = self.imread(img_A1)
                img_A2 = self.imread(img_A2)
                img_B = self.imread(img_B)
                img_A = np.stack([img_A1,img_A2])

                img_A = np.reshape(img_A,self.img_shapeA)
                img_B = np.reshape(cv2.resize(img_B, self.img_res),self.img_shapeB)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return cv2.imread(path, 0)
