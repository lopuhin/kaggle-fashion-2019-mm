import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    state = torch.load(args.rouce, map_location='cpu')
    for i in range(3):
        del state['state_dict'][f'bbox_head.{i}.fc_cls.weight']
        del state['state_dict'][f'bbox_head.{i}.fc_cls.bias']
    state['meta']['iter'] = 0
    state['meta']['epoch'] = 0
    torch.save(state, args.target)


if __name__ == '__main__':
    main()
