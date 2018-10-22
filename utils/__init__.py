# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 16:40
# @Author  : Tao
# @Project : ImageCaption
# @File    : __init__.py.py
"""
Use for 
"""

# ----- import package -----#
import argparse


# ----- Add extra function -----#


# ------------------------------#


def main(args):
    """
    main function
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Name', type=str,
                        default='None',
                        help='descript')
    args = parser.parse_args()
    main(args)
