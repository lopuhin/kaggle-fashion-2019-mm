import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    state = torch.load(args.source, map_location='cpu')
    for i in range(3):
        del state['state_dict'][f'bbox_head.{i}.fc_cls.weight']
        del state['state_dict'][f'bbox_head.{i}.fc_cls.bias']
        del state['state_dict'][f'mask_head.{i}.conv_logits.weight']
        del state['state_dict'][f'mask_head.{i}.conv_logits.bias']
    state['meta']['iter'] = 0
    state['meta']['epoch'] = 0
    torch.save(state, args.target)


if __name__ == '__main__':
    main()
