#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Main program for the Python Vehicle Simulator, which can be used
    to simulate and test guidance, navigation and control (GNC) systems.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd edition, John Wiley & Sons, Chichester, UK.
URL: https://www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import sys
import matplotlib.pyplot as plt

# from vehicles import *
# from lib import *
import vehicles
import lib

# Simulation parameters:
sampleTime = 0.02  # sample time [seconds]
N = 10000  # number of samples

# 3D plot and animation parameters where browser = {firefox,chrome,safari,etc.}
numDataPoints = 50  # number of 3D data points
FPS = 10  # frames per second (animated GIF)
filename = "3D_animation.gif"  # data file for animated GIF
browser = "edge"  # browser for visualization of animated GIF

###############################################################################
# Vehicle constructors
###############################################################################
lib.printSimInfo()

"""
DSRV('depthAutopilot',z_d)                                       
frigate('headingAutopilot',U,psi_d)
otter('headingAutopilot',psi_d,V_c,beta_c,tau_X)                  
ROVzefakkel('headingAutopilot',U,psi_d)                          
semisub('DPcontrol',x_d,y_d,psi_d,V_c,beta_c)                      
shipClarke83('headingAutopilot',psi_d,L,B,T,Cb,V_c,beta_c,tau_X)  
supply('DPcontrol',x_d,y_d,psi_d,V_c,beta_c)      
tanker('headingAutopilot',psi_d,V_c,beta_c,depth)    
remus100('depthHeadingAutopilot',z_d,psi_d,V_c,beta_c)             
torpedo('depthHeadingAutopilot',z_d,psi_d,V_c,beta_c)             

Call constructors without arguments to test step inputs, e.g. DSRV(), otter(), etc. 
"""

no = input("Please enter a vehicle no.: ")

match no:  #  The match statement requires Python >= 3.10
    case "1":
        vehicle = vehicles.DSRV("depthAutopilot", 60.0)
    case "2":
        vehicle = vehicles.frigate("headingAutopilot", 10.0, 100.0)
    case "3":
        vehicle = vehicles.otter("headingAutopilot", 100.0, 0.3, -30.0, 200.0)
    case "4":
        vehicle = vehicles.ROVzefakkel("headingAutopilot", 3.0, 100.0)
    case "5":
        vehicle = vehicles.semisub("DPcontrol", 10.0, 10.0, 40.0, 0.5, 190.0)
    case "6":
        vehicle = vehicles.shipClarke83("headingAutopilot", -20.0, 70, 8, 6, 0.7, 0.5, 10.0, 1e5)
    case "7":
        vehicle = vehicles.supply("DPcontrol", 4.0, 4.0, 50.0, 0.5, 20.0)
    case "8":
        vehicle = vehicles.tanker("headingAutopilot", -20, 0.5, 150, 20, 80)
    case "9":
        vehicle = vehicles.remus100("depthHeadingAutopilot", 30, 50, 1525, 0.5, 170)
    case "10":
        vehicle = vehicles.torpedo("depthHeadingAutopilot", 30, 50, 1525, 0.5, 170)
    case _:
        print("Error: Not a valid simulator option"), sys.exit()

lib.printVehicleinfo(vehicle, sampleTime, N)


###############################################################################
# Main simulation loop
###############################################################################
def main():

    [simTime, simData] = lib.simulate(N, sampleTime, vehicle)

    lib.plotVehicleStates(simTime, simData, 1)
    lib.plotControls(simTime, simData, vehicle, 2)
    lib.plot3D(simData, numDataPoints, FPS, filename, 3)

    """ Ucomment the line below for 3D animation in the web browswer. 
    Alternatively, open the animated GIF file manually in your preferred browser. """
    # webbrowser.get(browser).open_new_tab('file://' + os.path.abspath(filename))

    plt.show()
    plt.close()


main()
