# -*- coding: utf-8 -*-
"""
Conversion factors and units, in cgs. To convert a given value, in cgs, to the desired units,
divide by that unit.
Example:
    The speed of light in km·s⁻¹ would be
	c_km = c / km
"""
# length
cm = 1.
m = 1e2
km = 1e5
AU = 1.4959787066e13
ly = 9.460730472e17
pc = 3.0856776e18
kpc = 1e3 * pc
Mpc = 1e6 * pc
Gpc = 1e9 * pc
mm = 1e-1
micron = 1e-4
um = micron
nm = 1e-7
angstrom = 1e-8

# mass
Msun = 1.9891e33
g = 1.
kg = 1e3
mg = 1e-3

# time
s = 1.
hr = 3600.
yr_Sidereal = 3.1558145e7
yr_Tropical = 3.155692519e7
yr_Gregorian = 3.1556952e7
yr_Julian = 3.15576e7
yr = yr_Julian
Myr = 1e6 * yr
Gyr = 1e9 * yr

# energy
eV = 1.6021765e-12       # one electron-volt, in erg
keV = 1e3 * eV
J = 1e-7                 # one Joule, in erg