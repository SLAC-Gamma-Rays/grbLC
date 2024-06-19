#!/usr/bin/env python
#
# config.py
# Allow configuration options to be set.
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
# to `sfd` instead of `dustmaps`
#
import json
import os


class ConfigError(Exception):
    pass


class Configuration:
    """
    A class that stores the package configuration.
    By default, the configuration is loaded from
      ~/.sfdsrc
    This can be overridden by setting the environmental variable
    :obj:`SFD_CONFIG_FNAME`.
    Paths stored in the configuration file (such as the data
    directory, :obj:`data_dir`, can include environmental
    variables, which will be expanded.
    """

    def __init__(self, fname):
        self._success = False
        self.fname = fname
        self.load()

    def load(self):
        if os.path.isfile(self.fname):
            with open(self.fname) as f:
                try:
                    self._options = json.load(f)
                    self._success = True
                except ValueError as error:
                    print(('The config file appears to be corrupted:\n\n'
                           '    {fname}\n\n'
                           'Either fix the config file manually, or overwrite '
                           'it with a blank configuration as follows:\n\n'
                           '    from grblc.convert.sfd.config import config\n'
                           '    config.reset()\n\n'
                           'Note that this will delete your configuration! For '
                           'example, if you have specified a data directory, '
                           'then sfd will forget about its location.'
                          ).format(fname=self.fname))
                    self._options = {}
        else:
            print(('Configuration file not found:\n\n'
                   '    {fname}\n\n'
                   'To create a new configuration file in the default '
                   'location, run the following python code:\n\n'
                   '    from grblc.convert.sfd.config import config\n'
                   '    config.reset()\n\n'
                   'Note that this will delete your configuration! For '
                   'example, if you have specified a data directory, '
                   'then sfd will forget about its location.'
                  ).format(fname=self.fname))
            self._options = {}
            self._success = True

    def save(self, force=False):
        """
        Saves the configuration to a JSON, in the standard config location.
        Args:
            force (Optional[:obj:`bool`]): Continue writing, even if the original
                config file was not loaded properly. This is dangerous, because
                it could cause the previous configuration options to be lost.
                Defaults to :obj:`False`.
        Raises:
            :obj:`ConfigError`: if the configuration file was not successfully
                                loaded on initialization of the class, and
                                :obj:`force` is :obj:`False`.
        """
        if (not self._success) and (not force):
            raise ConfigError((
                'The config file appears to be corrupted:\n\n'
                '    {fname}\n\n'
                'Before attempting to save the configuration, please either '
                'fix the config file manually, or overwrite it with a blank '
                'configuration as follows:\n\n'
                '    from grblc.convert.sfd.config import config\n'
                '    config.reset()\n\n'
                ).format(fname=self.fname))

        with open(self.fname, 'w') as f:
            json.dump(self._options, f, indent=2)

    def __setitem__(self, key, value):
        self._options[key] = value
        self.save()

    def __getitem__(self, key):
        return self._options.get(key, None)

    def get(self, key, default=None):
        """
        Gets a configuration option, returning a default value if the specified
        key isn't set.
        """
        return self._options.get(key, default)

    def remove(self, key):
        """
        Deletes a key from the configuration.
        """
        self._options.pop(key, None)
        self.save()

    def reset(self):
        """
        Resets the configuration, and overwrites the existing configuration
        file.
        """
        self._options = {}
        self.save(force=True)
        self._success = True


# The package configuration filename
default_config_fname = os.path.expanduser('~/.sfdsrc')
config_fname = os.environ.get('SFD_CONFIG_FNAME', default_config_fname)
if default_config_fname != config_fname:
    print(f'Overriding default configuration file with {config_fname}')

# The package configuration. By default, this is read from ``~/.sfdsrc``.
# The default location can be overridden by setting the ``SFD_CONFIG_FNAME``
# environment variable.
#
# This is the object that the user should interact with in order to change
# settings. For example, to set the directory where large files (e.g., dust maps)
# will be stored:
#
# .. code-block:: python
#
#     from sfd.config import config
#     config['data_dir'] = '/path/to/data/directory'
config = Configuration(config_fname)
