from __future__ import absolute_import, division, print_function

import inspect
import os

from . import Footprint

#footprint_path = os.path.split(inspect.getfile(Survey))[0]
# for now. Have to modify by hand (!)
footprint_path = os.path.join(
    os.environ['GIT'], 'astro', 'astro', 'footprint', 'footprints')

"""AdvACT"""
AdvACT = Footprint(
    'AdvACT', filename=os.path.join(footprint_path, 'AdvACTSurveyMask_v3.reg'),
    default_plot_wrap=180)
_fp = AdvACT.footprint
for i in (1, 2):
    AdvACT.footprint[i][:,0][_fp[i][:,0] > 180] = \
        _fp[i][:,0][_fp[i][:,0] > 180] - 2*180

"""DES"""
DES = Footprint(
    'DES', filename=os.path.join(footprint_path, 'desfootprint.csv'))
