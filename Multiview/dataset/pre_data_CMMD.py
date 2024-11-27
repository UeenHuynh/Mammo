import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
from collections import Counter

class Dataset():
    def __init__(self, input_size, root, dataset_path, mode='train'):
        self.input_size = input_size
        self.root = root
        self.mode = mode

        # train_img_path = val_img_path = test_img_path = dataset_path
        self.train_img_path = self.test_img_path = dataset_path        

        train_label_file = open(os.path.join(self.root, 'cls_train_CMMD.txt'))
        # val_label_file = open(os.path.join(self.root,'val.txt'))
        test_label_file = open(os.path.join(self.root, 'cls_test_CMMD.txt'))

        train_img_label = []
        # val_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(self.train_img_path,line[:-1].split(' ')[0]), line[:-1].split(' ')[1]])
        # for line in val_label_file:
        #     val_img_label.append([os.path.join(val_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])
        for line in test_label_file:
            test_img_label.append([os.path.join(self.test_img_path,line[:-1].split(' ')[0]), line[:-1].split(' ')[1]])

        self.train_img_label = train_img_label
        # self.val_img_label = val_img_label
        self.test_img_label = test_img_label

        # count = [i[1] for i in self.train_img_label]

        # print(Counter(count).keys()) # equals to list(set(words))
        # print(Counter(count).values()) # counts the elements' frequency
        
        #create batch of 4 views
        train_img_label_multiview = {}
        test_img_label_multiview = {}

        # self.path_data = self.train_img_label[0][0][:-69]
        views = {'1': 'LCC',
                '2': 'LMLO',
                '3': 'RCC',
                '4': 'RMLO'
                }

        for i in range(len(self.train_img_label)):
            data_path = self.train_img_label[i][0]
            patient = data_path[-15:-6]
            view = data_path[-5]
            
            patientE = f'{patient}E'
            patientO = f'{patient}O'

            if patientE not in train_img_label_multiview or patientO not in train_img_label_multiview:
                train_img_label_multiview[patientE] = {}
                train_img_label_multiview[patientO] = {}
                #append image id and label 
                if (int(view)-1) // 2 == 0:
                    train_img_label_multiview[patientE][views[view]] = [view, self.train_img_label[i][1]]
                elif (int(view)-1) // 2 == 1:
                    train_img_label_multiview[patientO][views[view]] = [view, self.train_img_label[i][1]]
            else:
                if (int(view)-1) // 2 == 0:
                    train_img_label_multiview[patientE][views[view]] = [view, self.train_img_label[i][1]]
                elif (int(view)-1) // 2 == 1:
                    train_img_label_multiview[patientO][views[view]] = [view, self.train_img_label[i][1]]
            
        for i in range(len(self.test_img_label)):
            data_path = self.test_img_label[i][0]
            patient = data_path[-15:-6]
            view = data_path[-5]

            patientE = f'{patient}E'
            patientO = f'{patient}O'

            if patientE not in test_img_label_multiview or patientO not in test_img_label_multiview:
                test_img_label_multiview[patientE] = {}
                test_img_label_multiview[patientO] = {}
                #append image id and label 
                if (int(view)-1) // 2 == 0:
                    test_img_label_multiview[patientE][views[view]] = [view, self.test_img_label[i][1]]
                elif (int(view)-1) // 2 == 1:
                    test_img_label_multiview[patientO][views[view]] = [view, self.test_img_label[i][1]]
            else:
                if (int(view)-1) // 2 == 0:
                    test_img_label_multiview[patientE][views[view]] = [view, self.test_img_label[i][1]]
                elif (int(view)-1) // 2 == 1:
                    test_img_label_multiview[patientO][views[view]] = [view, self.test_img_label[i][1]]

        viewsL = ['LCC', 'LMLO']
        viewsR = ['RMLO', 'RCC']
        
        not_suitable = []
        for i in train_img_label_multiview: 
            if (set(viewsL) != set(train_img_label_multiview[i]) and set(viewsR) != set(train_img_label_multiview[i])) or len(list(train_img_label_multiview[i])) != 2:
                not_suitable.append(i)

        for i in not_suitable:
            train_img_label_multiview.pop(i)

        not_suitable = [] 
        for i in test_img_label_multiview: 
            if (set(viewsL) != set(test_img_label_multiview[i]) and set(viewsR) != set(test_img_label_multiview[i])) or len(list(test_img_label_multiview[i])) != 2:
                not_suitable.append(i)
        
        for i in not_suitable:
            test_img_label_multiview.pop(i)

        self.train_img_label_multiview = train_img_label_multiview
        self.test_img_label_multiview = test_img_label_multiview
            

    def preprocess(self, img):
        if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize(self.input_size + 16)(img)  # old 16
        img = transforms.RandomRotation(20)(img)
        img = transforms.RandomVerticalFlip()(img)
        img = transforms.RandomCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img 

    def __getitem__(self, index):
        labels = {'benign': 1, 
                  'malignancy' : 2,
        }        
        if self.mode == 'train':
            study = list(self.train_img_label_multiview)[index]
            if study[-1] == "E":
                cc_id, target = self.train_img_label_multiview[study]["LCC"]
                mlo_id = self.train_img_label_multiview[study]["LMLO"][0]
            else:
                cc_id, target = self.train_img_label_multiview[study]["RCC"]
                mlo_id = self.train_img_label_multiview[study]["RMLO"][0]

        else:
            study = list(self.test_img_label_multiview)[index]
            if study[-1] == "E":
                cc_id, target = self.test_img_label_multiview[study]["LCC"]
                mlo_id = self.test_img_label_multiview[study]["LMLO"][0]
            else:
                cc_id, target = self.test_img_label_multiview[study]["RCC"]
                mlo_id = self.test_img_label_multiview[study]["RMLO"][0]
        target = labels[target]
        study = study[:-1]
        img_cc =  self.preprocess(imageio.imread(f'{self.train_img_path}/{study}-{cc_id}.png'))
        img_mlo = self.preprocess(imageio.imread(f'{self.train_img_path}/{study}-{mlo_id}.png'))

        return img_cc, img_mlo, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_label_multiview)
        # elif self.mode == 'val':
        #     return len(self.val_img_label)
        else:
            return len(self.test_img_label_multiview)