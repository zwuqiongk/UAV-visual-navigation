import time
import math
import sys
import cv2
import UE4CtrlAPI
ue = UE4CtrlAPI.UE4CtrlAPI()

import PX4MavCtrlV4 as PX4MavCtrl


#Create a new MAVLink communication instance, UDP sending port (CopterSim’s receving port) is 20100
mav = PX4MavCtrl.PX4MavCtrler(1)


# sendUE4Cmd: RflySim3D API to modify scene display style
# Format: ue.sendUE4Cmd(cmd,windowID=-1), where cmd is a command string, windowID is the received window number (assuming multiple RflySim3D windows are opened at the same time), windowID =-1 means sent to all windows
# Augument: RflyChangeMapbyName command means to switch the map (scene), the following string is the map name, here will switch all open windows to the grass map
ue.sendUE4Cmd('RflyChangeMapbyName Grasslands')
time.sleep(2)

# Create a vehicle and setting the ground height and init PosX, posY, and Yaw angle
mav.initPointMassModel(-8.086,[0,0,0])

time.sleep(2)
mav.SendVelNED(0,0,-2,0) # takeoff with speed 2m/s

time.sleep(6)
mav.SendVelNED(3,0,0,0) # fly north (X) with speed 3m/s

time.sleep(6)
mav.SendVelNED(0,0,0,0.1) # stay hold and rotate with yawrate 0.1rad/s/s


time.sleep(6)
mav.SendVelFRD(3,0,0,0) # fly forward (body frame) with speed 3m/s


time.sleep(6)
mav.SendPosNED(5,5,-10,0) # fly to position 5,5,-10 (Earth frame)


time.sleep(10)
mav.SendPosNED(0,0,0,0) # fly to position 0,0,0 (Earth frame)


time.sleep(10)
mav.EndPointMassModel() # End simulation
