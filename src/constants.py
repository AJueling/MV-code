""" model grid constants """

imt, jmt, km = 3600, 2400, 42     # POP high resolution grid dimensions


""" physical constants (SI units) """

g                 = 9.81          # [m s^-2]         gravitional acceleration
sigma             = 5.670e-8      # [W m^−2 K^−4]    Stefan-Boltzmann constant
rho_sw            = 1026.         # [kg m^-3]        reference density of sea water
cp_sw             = 3996.         # [J kg^-1 K^-1]   heat capacity of sea water
cp_air            = 1005.         # [J kg^-1 K^-1]   heat capacity of dry air at constant pressure
                                  #                  (approx. constant for atm. pressure range)
R_earth           = 6.371e6       # [m]              radius of the Earth
A_earth           = 5.101e14      # [m^2]            surface area of the Earth
abs_zero          = -273.15       # [K]              absolute freezing
latent_heat_vapor = 2501000.      # [J kg^-1]        latent heat of vaporization


""" unit conversions """

spy               = 3600*24*365   # [s yr^-1]
spd               = 3600*24       # [s day^-1]
Jpy_to_Wpsm       = 1/spy/A_earth # [J/yr] to [W/m^2]
