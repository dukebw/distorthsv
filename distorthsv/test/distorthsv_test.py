# Copyright 2018 Brendan Duke.
#
# This file is part of distorthsv.
#
# distorthsv is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# distorthsv is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# distorthsv. If not, see <http://www.gnu.org/licenses/>.

"""Unit test for distorthsv."""
import multiprocessing
import time

import click
import numpy as np
import matplotlib.pyplot as plt

import distorthsv
import loadvid


@click.command()
@click.option('--filename',
              default=None,
              type=str,
              help='Name of the input video.')
@click.option('--width',
              default=None,
              type=int,
              help='The _exact_ width of the input video.')
@click.option('--height',
              default=None,
              type=int,
              help='The _exact_ height of the input video.')
def distorthsv_test(filename, width, height):
    """Tests distorthsv Python extension."""
    with open(filename, 'rb') as f:
        encoded_video = f.read()

    num_frames = 32
    for i in range(20):
        decoded_frames, _ = loadvid.loadvid(encoded_video,
                                            should_random_seek=True,
                                            width=width,
                                            height=height,
                                            num_frames=num_frames)
        decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
        decoded_frames = np.reshape(decoded_frames,
                                    newshape=(num_frames, height, width, 3))
        decoded_frames = decoded_frames.astype(np.float32)
        decoded_frames /= 255.

        start = time.perf_counter()

        saturation_factor = 1.0
        hue_factor = 1.0
        brightness_factor = 1.0
        contrast_factor = 1.0
        if i < 5:
            hue_factor = 0.5 + i/4
        elif 5 <= i < 10:
            saturation_factor = 0.5 + (i - 5)/4
        elif 10 <= i < 15:
            brightness_factor = 0.5 + (i - 10)/4
        else:
            contrast_factor = 0.5 + (i - 15)/4

        distorthsv.distorthsv(decoded_frames,
                              hue_factor=hue_factor,
                              saturation_factor=saturation_factor,
                              brightness_factor=brightness_factor,
                              max_num_threads=multiprocessing.cpu_count())

        distorthsv.distort_contrast(
            decoded_frames,
            contrast_factor=contrast_factor,
            max_num_threads=multiprocessing.cpu_count())

        distorthsv.fliplr(decoded_frames,
                          max_num_threads=multiprocessing.cpu_count())

        end = time.perf_counter()

        print('time: {}'.format(end - start))

        plt.imshow(decoded_frames[0, ...])
        plt.show()
