#!/usr/bin/env python
#
# std_paths.py
# Defines a set of paths used by scripts in the sfd module.
#
# Copyright (C) 2016  Gregory M. Green
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# This file has been modified by S. Young to change naming
# to `sfd` instead of `dustmaps`. Additionally, the `output` functionality
# is vestigial with respect to our usage of `dustmaps`, so it has been
# removed. Data and test directories have been modified as appropriately.
#
import os

from .config import config


script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
up_one_dir = os.path.abspath(os.path.join(script_dir, "../"))
data_dir_default = os.path.abspath(os.path.join(up_one_dir, 'extinction_maps'))
test_dir = os.path.abspath(os.path.join(parent_dir, 'tests'))


def fix_path(path):
    """
    Returns an absolute path, expanding both '~' (to the user's home
    directory) and other environmental variables in the path.
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def data_dir():
    """
    Returns the directory used to store large data files (e.g., dust maps).
    """
    dirname = config.get('data_dir', data_dir_default)
    return fix_path(dirname)
