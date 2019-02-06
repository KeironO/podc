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

import os
from .video import get_max_frames
from .models import *

def get_labels(d):
    names = ["positive", "negative"]
    labels_dict = {}
    for n in names:
        fn = "videos of %s sliding sign from reproducibility study" % (n)
        files = os.listdir(os.path.join(d, fn))
        if n == "negative":
            n = 0
        else:
            n = 1
        for i in files:
            labels_dict[i] = n

    return labels_dict