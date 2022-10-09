import argparse
from tools import create_dataset_files

def get_args():
    parser = argparse.ArgumentParser(description='prepare')
    parser.add_argument('--dataset', type=str, default='flickr8k', help='dataset name')
    parser.add_argument('--resize', type=int, default=256, help='resize size')
    parser.add_argument('--json_path', type=str, default='mnt/disk3/quyi/data/Flickr8k/caption_datasets/dataset_flickr8k.json', help='josn path')
    parser.add_argument('--caption_per_image', type=int, default=5, help='caption num per image')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    create_dataset_files(dataset='flickr8k',
                         karpathy_json_path='mnt/disk3/quyi/data/Flickr8k/')