from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader

class SequentialLoader(object):

    def get_transform(self):
        train_transform = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(),
            RandomCrop(224, padding=4),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        test_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        return train_transform, test_transform

    def get_class(self, args):
        '''
        assume classes across domains are the same.
        [0 1 ..................................................................... N - 1]
        |----common classes --||----source private classes --||----target private classes --|
        '''
        a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
        c = c - a - b
        common_classes = [i for i in range(a)]
        source_private_classes = [i + a for i in range(b)]
        target_private_classes = [i + a + b for i in range(c)]

        source_classes = common_classes + source_private_classes
        target_classes = common_classes + target_private_classes
        return source_classes, target_classes



    def split(self, train, val, val_size=0.1):
        train_imgs, train_labels, val_imgs, val_labels = [], [], [], []
        for t in list(set(train.dataset.labels)):
            index = list(np.where(np.array(train.dataset.labels) == t))[0]
            num_of_exapmples = int(val_size * len(index))
            import random
            random.shuffle(index)
            train_imgs.extend(np.array(train.dataset.datas)[index[:]].tolist())
            train_labels.extend(np.array(train.dataset.labels)[index[:]].tolist())
            val_imgs.extend(np.array(train.dataset.datas)[index[-num_of_exapmples:]].tolist())
            val_labels.extend(np.array(train.dataset.labels)[index[-num_of_exapmples:]].tolist())
        train.dataset.datas, train.dataset.labels = train_imgs, train_labels
        val.dataset.datas, val.dataset.labels = val_imgs, val_labels


    def __init__(self, dataset, args, episode_length=10, val_size=0.1, val_flag=False):
        # get class label
        self.source_classes, self.target_classes = self.get_class(args)
        # get image transform
        self.train_transform, self.test_transform = self.get_transform()
        # load source data
        self.sources, self.targets, self.episode_length, self.val_flag = [], [], episode_length, val_flag
        for i in range(episode_length):
            ds = FileListDataset(list_path=dataset.files[i], path_prefix=dataset.prefixes[args.data.dataset.source], transform=self.train_transform, filter=(lambda x: x in self.source_classes))
            dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=args.data.dataloader.data_workers, drop_last=True)
            self.sources.append(dl)
        # load source validation data
        self.sources_val = []
        for i in range(episode_length):
            ds = FileListDataset(list_path=dataset.files[i], path_prefix=dataset.prefixes[args.data.dataset.source], transform=self.train_transform, filter=(lambda x: x in self.source_classes))
            dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=args.data.dataloader.data_workers, drop_last=True)
            self.sources_val.append(dl)
        for i in range(episode_length):

            self.split(self.sources[i], self.sources_val[i], val_size)

        # load target data
        for i in range(episode_length, episode_length + episode_length):
            ds = FileListDataset(list_path=dataset.files[i], path_prefix=dataset.prefixes[args.data.dataset.target], transform=self.train_transform, filter=(lambda x: x in self.target_classes))
            dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=args.data.dataloader.data_workers, drop_last=True)
            self.targets.append(dl)

        # load test data
        ds = FileListDataset(list_path=dataset.files[episode_length + episode_length], path_prefix=dataset.prefixes[args.data.dataset.target], transform=self.test_transform, filter=(lambda x: x in self.target_classes))
        self.target_test = DataLoader(dataset=ds, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
        ds = FileListDataset(list_path=dataset.files[episode_length + episode_length+1], path_prefix=dataset.prefixes[args.data.dataset.target], transform=self.test_transform, filter=(lambda x: x in self.target_classes))
        self.source_test = DataLoader(dataset=ds, batch_size=32, shuffle=False, num_workers=1, drop_last=False)

    def __call__(self, i):
        return self.sources[i], self.sources_val[i], self.source_test, self.targets[i], self.target_test



def get_loader():
    loader = SequentialLoader(dataset, args, episode_length=10, val_size=args.misc.val_size, val_flag=bool(args.misc.val_flag))

    ds = FileListDataset(list_path=dataset.files[0], path_prefix=dataset.prefixes[args.data.dataset.source], transform=loader.train_transform, filter=(lambda x: x in loader.source_classes))
    # set memory
    memory_dl = DataLoader(dataset=ds, batch_size=args.data.dataloader.batch_size, shuffle=True, num_workers=args.data.dataloader.data_workers, drop_last=True)
    ds = FileListDataset(list_path=dataset.files[0], path_prefix=dataset.prefixes[args.data.dataset.source], transform=loader.train_transform, filter=(lambda x: x in loader.source_classes))
    memory_val_dl = DataLoader(dataset=ds, batch_size=args.data.dataloader.batch_size, shuffle=True, num_workers=args.data.dataloader.data_workers, drop_last=True)

    memory_dl.dataset.datas, memory_dl.dataset.labels = [], []
    memory_val_dl.dataset.datas, memory_val_dl.dataset.labels = [], []
    return loader, memory_dl, memory_val_dl


