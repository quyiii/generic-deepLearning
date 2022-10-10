import sys
import argparse
sys.path.append('.')
from tools import create_dataset_files

def get_args():
    parser = argparse.ArgumentParser(description='prepare')
    parser.add_argument('--dataset', type=str, default='flickr8k', help='dataset name')
    parser.add_argument('--json_path', type=str, default='/mnt/disk3/quyi/data/Flickr8k/caption_datasets/dataset_flickr8k.json', help='josn path')
    parser.add_argument('--image_folder', type=str, default='/mnt/disk3/quyi/data/Flickr8k/image', help='images path')
    parser.add_argument('--captions_per_image', type=int, default=5, help='caption num per image')
    parser.add_argument('--min_word_freq', type=int, default=5, help='min frequency of word')
    parser.add_argument('--output_folder', type=str, default='/mnt/disk3/quyi/data/Flickr8k/dataset', help='output folder of files')
    parser.add_argument('--max_len', type=int, default=50, help='max caption len')
    parser.add_argument('--resize', type=int, default=256, help='resize size')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    create_dataset_files(dataset=args.dataset,
                         karpathy_json_path=args.json_path,
                         image_folder=args.image_folder,
                         captions_per_image=args.captions_per_image,
                         min_word_freq=args.min_word_freq,
                         output_folder=args.output_folder,
                         max_len=args.max_len,
                         resize=args.resize)