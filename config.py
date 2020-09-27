import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file), Loader=yaml.SafeLoader)

save_config = yaml.load(open(config_file), Loader=yaml.SafeLoader)

args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'cosda':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['train_source', 'train_target', 'test_target'],
    files=[
        './sequential_source/source_list_0.txt', # 0
        './sequential_source/source_list_1.txt', # 1
        './sequential_source/source_list_2.txt', # 2
        './sequential_source/source_list_3.txt', # 3
        './sequential_source/source_list_4.txt', # 4
        './sequential_source/source_list_5.txt', # 5
        './sequential_source/source_list_6.txt', # 6
        './sequential_source/source_list_7.txt', # 7
        './sequential_source/source_list_8.txt', # 8
        './sequential_source/source_list_9.txt', # 9
        './sequential_target/target_list_0.txt', # 10
        './sequential_target/target_list_1.txt', # 10
        './sequential_target/target_list_2.txt', # 11
        './sequential_target/target_list_3.txt', # 12
        './sequential_target/target_list_4.txt', # 13
        './sequential_target/target_list_5.txt', # 14
        './sequential_target/target_list_6.txt', # 15
        './sequential_target/target_list_7.txt', # 16
        './sequential_target/target_list_8.txt', # 17
        './sequential_target/target_list_9.txt', # 18
        './eval/evaluation_list.txt', # 20
        './eval/source_eval_list.txt', # 21
    ],
    prefix=args.data.dataset.root_path)
    dataset.prefixes = [dataset.path, dataset.path]
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
target_val_domain_name = dataset.domains[args.data.dataset.target_val]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
target_val_file = dataset.files[args.data.dataset.target_val]
