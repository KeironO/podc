'''
Copyright (c) 2019 Keiron O'Shea

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301 USA
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def visualise_video_data(X):
    fig = plt.figure()

    frames = []

    print(X.shape)
    for index in range(X.shape[0]):
        frames.append([plt.imshow(X[index]/255)])
    ani = animation.ArtistAnimation(fig, frames)
    plt.show()
    
