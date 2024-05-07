import astropy.units as u

class customUnitsClass():
    def __init__(self):
        self.mps = 1 * u.m/u.s
        self.kmph = 1 * u.km / u.hour
        self.mps2 = (1 * u.m) / (u.s*u.s)
        self.gconst = 1 * (u.m * u.m * u.m)/(u.kg * u.s * u.s)


