"""Reads and writes the configuration file of PyARTOS.

This module provides access to the configuration given in the file 'pyartos.ini',
which is searched for in the current working directory. To access the configuration
options, the 'config' object in this module's dictionary can be used, which is an
instance of the private _Config class.
"""

try:
    # Python 3
    from configparser import ConfigParser as SafeConfigParser
except:
    # Python 2
    from ConfigParser import SafeConfigParser

import os.path


class _Config(SafeConfigParser):
    """Class to access configuration options of PyARTOS.
    
    Rather than instantiating this class, use the 'config' instance available in the dictionary
    of the module.
    """


    def __init__(self, iniFile):
        """Initializes a new configuration instance and sets appropriate default values."""
        
        SafeConfigParser.__init__(self, allow_no_value = True)
        self.iniFileName = iniFile
        self.read(iniFile)
        self.defaults = { }
        self.applyDefaults({
            'libartos' : {
                'model_dir'             : _Config.findModelDir(),
                'library_path'          : None,
                'debug'                 : 0,
            },
            'ImageNet' : {
                'repository_directory'  : None
            },
            'GUI' : {
                'max_video_width'       : 640,
                'max_video_height'      : 480
            }
        });


    def applyDefaults(self, defaultDict):
        """Applies default values from a dictionary.
        
        defaultDict - Dictionary whose keys are section names and whose values are dictionaries
                      with the default configuration options for that section.
        """
        
        for section in defaultDict:
            if not section in self.defaults:
                self.defaults[section] = { }
            if not self.has_section(section):
                self.add_section(section)
            for option in defaultDict[section]:
                self.defaults[section][option] = defaultDict[section][option]
                if not self.has_option(section, option):
                    self.set(section, option, None)
    
    
    def get(self, section, option, useDefaults = True):
        """Get an option value for the named section.
        
        If useDefaults is set to true, this function falls back to default values
        if the given option hasn't been specified in the configuration file or is empty.
        """
        
        try:
            value = SafeConfigParser.get(self, section, option)
        except:
            value = None
        if useDefaults and ((value is None) or (value == '')) \
                and (section in self.defaults) and (option in self.defaults[section]):
            value = self.defaults[section][option]
        return value
    
    
    def getInt(self, section, option, useDefaults = True, min = None, max = None):
        """Get an option value for the named section as integer.
        
        If useDefaults is set to true, this function falls back to default values if the
        given option hasn't been specified in the configuration file or isn't an integral value.
        The range of allowable values can be specified using the min and max parameters. The value
        from the configuration file will be clipped to that range.
        """
        
        try:
            value = int(SafeConfigParser.get(self, section, option))
        except:
            value = None
        if useDefaults and ((value is None) or (value == '')) \
                and (section in self.defaults) and (option in self.defaults[section]):
            value = self.defaults[section][option]
        if not value is None:
            if (not min is None) and (value < min):
                value = min
            elif (not max is None) and (value > max):
                value = max
        return value
    
    
    def getBool(self, section, option, useDefaults = True):
        """Get an option value for the named section as boolean value.
        
        If useDefaults is set to true, this function falls back to default values if the
        given option hasn't been specified in the configuration file or can't be interpreted as a boolean value.
        Empty strings, the strings 'no', 'off' and 'false' as well as the number 0 will be interpreted as False.
        Every number different from 0 as well as the strings 'yes', 'on' and 'true' will be interpreted as True.
        """
        
        def toBool(str):
            try:
                intval = int(str)
                return (intval != 0)
            except:
                pass
            try:
                if str.lower() in ('', 'no', 'off', 'false'):
                    return False
                elif str.lower() in ('yes', 'on', 'true'):
                    return True
            except:
                pass
            return None
        
        value = toBool(SafeConfigParser.get(self, section, option))
        if useDefaults and ((value is None) or (value == '')) \
                and (section in self.defaults) and (option in self.defaults[section]):
            value = toBool(self.defaults[section][option])
        return value
    
    
    def is_set(self, section, option):
        """Determines if a given option has been set in the configuration file (regardless of default values)."""
        
        try:
            value = SafeConfigParser.get(self, section, option)
        except:
            value = None
        return (not value is None) and (value != '')
    
    
    def differentFromDefault(self, section, option):
        """Determines if a given configuration option differs from it's default value."""
        
        if (not section in self.defaults) or (not option in self.defaults[section]):
            return True
        else:
            return (self.get(section, option) != self.defaults[section][option])


    def save(self):
        """Writes the configuration options to the file they were read from."""
        
        with open(self.iniFileName, 'w') as file:
            self.write(file)


    @staticmethod
    def findModelDir():
        """Searches for a potential model directory.
        
        Searches for a directory named 'models' in the current working directory, one level above
        the current working directory, the executed script's directory and the packages's directory.
        The first match is returned or an empty string if no directory has been found.
        """
        
        basedir = os.path.dirname(os.path.abspath(__file__))
        tests = ['models', os.path.join('..','models'), os.path.join(basedir,'..','models'), os.path.join(basedir,'models')]
        for t in tests:
            if (os.path.isdir(t)):
                return os.path.realpath(t)
        else:
            return ''



config = _Config('pyartos.ini')