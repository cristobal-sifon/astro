from __future__ import absolute_import, division, print_function

import inspect
import os

from . import Footprint

footprint_path = os.path.split(inspect.getfile(Footprint))[0]
footprint_path = os.path.join(footprint_path, 'footprints')


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


"""SPLUS survey"""
SPLUS = Footprint(
    'SPLUS', filename=os.path.join(footprint_path, 'S-PLUS_footprint.csv'))