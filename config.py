# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 16:40
# @Author  : Tao
# @Project : ImageCaption
# @File    : config.py
"""
Use for 
"""

# ----- import package -----#
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int,
                        default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='data/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int,
                        default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int,
                        default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int,
                        default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int,
                        default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    return args

