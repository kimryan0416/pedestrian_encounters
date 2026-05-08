# =================================
# Imports
# =================================

import numpy as np
import pandas as pd
from collections import Counter

# =================================
# Global Variables
# =================================

_NAME_CATEGORY_DICT = {
    'NorthSidewalk': ['Environment', 'Sidewalk', 'North', 'NorthSidewalk'], 
    'SouthSidewalk': ['Environment', 'Sidewalk', 'South', 'SouthSidewalk'],
    'RoadEast': ['Environment', 'Road', 'East', 'RoadEast'], 
    'RoadWest': ['Environment', 'Road', 'West', 'RoadWest'],
    'RoadCrosswalk': ['Environment', 'Road', 'Crosswalk', 'RoadCrosswalk'],
    'NorthWalkingPole': ['Environment', 'Pole', 'North', 'NorthWalkingPole'], 
    'NorthDiagonalPole': ['Environment', 'Pole', 'North', 'NorthDiagonalPole'],
    'SouthWalkingPole': ['Environment', 'Pole', 'South', 'SouthWalkingPole'],
    'SouthDiagonalPole': ['Environment', 'Pole', 'South', 'SouthDiagonalPole'],
    'NorthCarSignal': ['Environment', 'Car_Signal', 'North', 'NorthCarSignal'],
    'SouthCarSignal': ['Environment', 'Car_Signal', 'South', 'SouthCarSignal'],
    'NorthWalkingSignal': ['Environment', 'Walking_Signal', 'North', 'NorthWalkingSignal'],
    'SouthWalkingSignal': ['Environment', 'Walking_Signal', 'South', 'SouthWalkingSignal'],
    'NorthBuildings_NoColliders': ['Environment', 'Buildings', 'North', 'NorthBuildings'],
    'SouthBuildings_NoColliders': ['Environment', 'Buildings', 'South', 'SouthBuildings'], 
    'NE_Tree_10': ['Environment', 'Tree', 'NE', 'NE_Tree_10'], 
    'NE_Tree_30': ['Environment', 'Tree', 'NE', 'NE_Tree_30'], 
    'NE_Tree_50': ['Environment', 'Tree', 'NE', 'NE_Tree_50'], 
    'NE_Tree_70': ['Environment', 'Tree', 'NE', 'NE_Tree_70'],
    'NW_Tree_10': ['Environment', 'Tree', 'NW', 'NW_Tree_10'], 
    'NW_Tree_30': ['Environment', 'Tree', 'NW', 'NW_Tree_30'], 
    'NW_Tree_50': ['Environment', 'Tree', 'NW', 'NW_Tree_50'], 
    'NW_Tree_70': ['Environment', 'Tree', 'NW', 'NW_Tree_70'],
    'SE_Tree_10': ['Environment', 'Tree', 'SE', 'SE_Tree_10'], 
    'SE_Tree_30': ['Environment', 'Tree', 'SE', 'SE_Tree_30'], 
    'SE_Tree_50': ['Environment', 'Tree', 'SE', 'SE_Tree_50'], 
    'SE_Tree_70': ['Environment', 'Tree', 'SE', 'SE_Tree_70'],
    'SW_Tree_10': ['Environment', 'Tree', 'SW', 'SW_Tree_10'], 
    'SW_Tree_30': ['Environment', 'Tree', 'SW', 'SW_Tree_30'], 
    'SW_Tree_50': ['Environment', 'Tree', 'SW', 'SW_Tree_50'], 
    'SW_Tree_70': ['Environment', 'Tree', 'SW', 'SW_Tree_70'],
    'ApproachAgent': ['Pedestrian', 'Confederate', 'Approach', 'ApproachAgent'], 
    'BehindRunner': ['Pedestrian', 'Confederate', 'Behind', 'BehindRunner'], 
    'AlleywayRunner': ['Pedestrian', 'Confederate', 'Alleyway', 'AllewayRunner'],
    '0': ['Pedestrian', 'Bystander', '', '0'], 
    '1': ['Pedestrian', 'Bystander', '', '1'], 
    '2': ['Pedestrian', 'Bystander', '', '2'], 
    '3': ['Pedestrian', 'Bystander', '', '3'], 
    '4': ['Pedestrian', 'Bystander', '', '4'], 
    '5': ['Pedestrian', 'Bystander', '', '5'], 
    '6': ['Pedestrian', 'Bystander', '', '6'], 
    '7': ['Pedestrian', 'Bystander', '', '7'], 
    '8': ['Pedestrian', 'Bystander', '', '8'], 
    '9': ['Pedestrian', 'Bystander', '', '9'],
    '10': ['Pedestrian', 'Bystander', '', '10'], 
    '11': ['Pedestrian', 'Bystander', '', '11'], 
    '12': ['Pedestrian', 'Bystander', '', '12'], 
    '13': ['Pedestrian', 'Bystander', '', '13'], 
    '14': ['Pedestrian', 'Bystander', '', '14'], 
    '15': ['Pedestrian', 'Bystander', '', '15'], 
    '16': ['Pedestrian', 'Bystander', '', '16'], 
    '17': ['Pedestrian', 'Bystander', '', '17'], 
    '18': ['Pedestrian', 'Bystander', '', '18'], 
    '19': ['Pedestrian', 'Bystander', '', '19'],
    '20': ['Pedestrian', 'Bystander', '', '20'], 
    '21': ['Pedestrian', 'Bystander', '', '21'], 
    '22': ['Pedestrian', 'Bystander', '', '22'], 
    '23': ['Pedestrian', 'Bystander', '', '23'], 
    '24': ['Pedestrian', 'Bystander', '', '24'], 
    '25': ['Pedestrian', 'Bystander', '', '25'], 
    '26': ['Pedestrian', 'Bystander', '', '26'], 
    '27': ['Pedestrian', 'Bystander', '', '27'], 
    '28': ['Pedestrian', 'Bystander', '', '28'], 
    '29': ['Pedestrian', 'Bystander', '', '29'],
    'Vehicle.Car.Body-Car1.Body':       ['Vehicle', 'Car', 'Car 1', 'Body'],
    'Vehicle.Car.Door-Car1.Door.FR':    ['Vehicle', 'Car', 'Car 1', 'FR Door'],
    'Vehicle.Car.Door-Car1.Door.FL':    ['Vehicle', 'Car', 'Car 1', 'FL Door'],
    'Vehicle.Car.Door-Car1.Door.BR':    ['Vehicle', 'Car', 'Car 1', 'BR Door'],
    'Vehicle.Car.Door-Car1.Door.BL':    ['Vehicle', 'Car', 'Car 1', 'BL Door'],
    'Vehicle.Car.Wheel-Car1.Wheel.FR':  ['Vehicle', 'Car', 'Car 1', 'FR Wheel'], 
    'Vehicle.Car.Wheel-Car1.Wheel.FL':  ['Vehicle', 'Car', 'Car 1', 'FL Wheel'], 
    'Vehicle.Car.Wheel-Car1.Wheel.BR':  ['Vehicle', 'Car', 'Car 1', 'BR Wheel'],
    'Vehicle.Car.Wheel-Car1.Wheel.BL':  ['Vehicle', 'Car', 'Car 1', 'BL Wheel'],
    'Vehicle.Car.Driver-Car1.Driver':   ['Vehicle', 'Car', 'Car 1', 'Driver'],
    'Vehicle.Car.Body-Car2.Body':       ['Vehicle', 'Car', 'Car 2', 'Body'],
    'Vehicle.Car.Door-Car2.Door.FR':    ['Vehicle', 'Car', 'Car 2', 'FR Door'],
    'Vehicle.Car.Door-Car2.Door.FL':    ['Vehicle', 'Car', 'Car 2', 'FL Door'],
    'Vehicle.Car.Door-Car2.Door.BR':    ['Vehicle', 'Car', 'Car 2', 'BR Door'],
    'Vehicle.Car.Door-Car2.Door.BL':    ['Vehicle', 'Car', 'Car 2', 'BL Door'],
    'Vehicle.Car.Wheel-Car2.Wheel.FR':  ['Vehicle', 'Car', 'Car 2', 'FR Wheel'], 
    'Vehicle.Car.Wheel-Car2.Wheel.FL':  ['Vehicle', 'Car', 'Car 2', 'FL Wheel'], 
    'Vehicle.Car.Wheel-Car2.Wheel.BR':  ['Vehicle', 'Car', 'Car 2', 'BR Wheel'],
    'Vehicle.Car.Wheel-Car2.Wheel.BL':  ['Vehicle', 'Car', 'Car 2', 'BL Wheel'],
    'Vehicle.Car.Driver-Car2.Driver':   ['Vehicle', 'Car', 'Car 2', 'Driver'],
    'Vehicle.Car.Body-Car3.Body':       ['Vehicle', 'Car', 'Car 3', 'Body'],
    'Vehicle.Car.Wheel-Car3.Body':      ['Vehicle', 'Car', 'Car 3', 'Body'],
    'Vehicle.Car.Door-Car3.Door.FR':    ['Vehicle', 'Car', 'Car 3', 'FR Door'],
    'Vehicle.Car.Door-Car3.Door.FL':    ['Vehicle', 'Car', 'Car 3', 'FL Door'],
    'Vehicle.Car.Door-Car3.Door.BR':    ['Vehicle', 'Car', 'Car 3', 'BR Door'],
    'Vehicle.Car.Door-Car3.Door.BL':    ['Vehicle', 'Car', 'Car 3', 'BL Door'],
    'Vehicle.Car.Wheel-Car3.Wheel.FR':  ['Vehicle', 'Car', 'Car 3', 'FR Wheel'], 
    'Vehicle.Car.Wheel-Car3.Wheel.FL':  ['Vehicle', 'Car', 'Car 3', 'FL Wheel'], 
    'Vehicle.Car.Wheel-Car3.Wheel.BR':  ['Vehicle', 'Car', 'Car 3', 'BR Wheel'],
    'Vehicle.Car.Wheel-Car3.Wheel.BL':  ['Vehicle', 'Car', 'Car 3', 'BL Wheel'],
    'Vehicle.Car.Driver-Car3.Driver':   ['Vehicle', 'Car', 'Car 3', 'Driver'],
    'Vehicle.Car.Body-Car4.Body':       ['Vehicle', 'Car', 'Car 4', 'Body'],
    'Vehicle.Car.Door-Car4.Door.FR':    ['Vehicle', 'Car', 'Car 4', 'FR Door'],
    'Vehicle.Car.Door-Car4.Door.FL':    ['Vehicle', 'Car', 'Car 4', 'FL Door'],
    'Vehicle.Car.Door-Car4.Door.BR':    ['Vehicle', 'Car', 'Car 4', 'BR Door'],
    'Vehicle.Car.Door-Car4.Door.BL':    ['Vehicle', 'Car', 'Car 4', 'BL Door'],
    'Vehicle.Car.Wheel-Car4.Wheel.FR':  ['Vehicle', 'Car', 'Car 4', 'FR Wheel'], 
    'Vehicle.Car.Wheel-Car4.Wheel.FL':  ['Vehicle', 'Car', 'Car 4', 'FL Wheel'], 
    'Vehicle.Car.Wheel-Car4.Wheel.BR':  ['Vehicle', 'Car', 'Car 4', 'BR Wheel'],
    'Vehicle.Car.Wheel-Car4.Wheel.BL':  ['Vehicle', 'Car', 'Car 4', 'BL Wheel'],
    'Vehicle.Car.Driver-Car4.Driver':   ['Vehicle', 'Car', 'Car 4', 'Driver'],
    'Vehicle.Car.Body-Car5.Body':       ['Vehicle', 'Car', 'Car 5', 'Body'],
    'Vehicle.Car.Door-Car5.Door.FR':    ['Vehicle', 'Car', 'Car 5', 'FR Door'],
    'Vehicel.Car.Door-Car5.Door.FR':    ['Vehicle', 'Car', 'Car 5', 'FR Door'],
    'Vehicle.Car.Door-Car5.Door.FL':    ['Vehicle', 'Car', 'Car 5', 'FL Door'],
    'Vehicle.Car.Door-Car5.Door.BR':    ['Vehicle', 'Car', 'Car 5', 'BR Door'],
    'Vehicle.Car.Door-Car5.Door.BL':    ['Vehicle', 'Car', 'Car 5', 'BL Door'],
    'Vehicle.Car.Wheel-Car5.Wheel.FR':  ['Vehicle', 'Car', 'Car 5', 'FR Wheel'], 
    'Vehicle.Car.Wheel-Car5.Wheel.FL':  ['Vehicle', 'Car', 'Car 5', 'FL Wheel'], 
    'Vehicle.Car.Wheel-Car5.Wheel.BR':  ['Vehicle', 'Car', 'Car 5', 'BR Wheel'],
    'Vehicle.Car.Wheel-Car5.Wheel.BL':  ['Vehicle', 'Car', 'Car 5', 'BL Wheel'],
    'Vehicle.Car.Driver-Car5.Driver':   ['Vehicle', 'Car', 'Car 5', 'Driver'],

    'Vehicle.Sedan.Body-Sedan1.Body':       ['Vehicle', 'Sedan', 'Sedan 1', 'Body'],
    'Vehicle.Sedan.Door-Sedan1.Door.FL':    ['Vehicle', 'Sedan', 'Sedan 1', 'FL Door'],
    'Vehicle.Sedan.Door-Sedan1.Door.FR':    ['Vehicle', 'Sedan', 'Sedan 1', 'FR Door'],
    'Vehicle.Sedan.Door-Sedan1.Door.BL':    ['Vehicle', 'Sedan', 'Sedan 1', 'BL Door'],
    'Vehicle.Sedan.Door-Sedan1.Door.BR':    ['Vehicle', 'Sedan', 'Sedan 1', 'BR Door'],
    'Vehicle.Sedan.Wheel-Sedan1.Wheel.FL':  ['Vehicle', 'Sedan', 'Sedan 1', 'FL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan1.Wheel.FR':  ['Vehicle', 'Sedan', 'Sedan 1', 'FR Wheel'],
    'Vehicle.Sedan.Wheel-Sedan1.Wheel.BL':  ['Vehicle', 'Sedan', 'Sedan 1', 'BL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan1.Wheel.BR':  ['Vehicle', 'Sedan', 'Sedan 1', 'BR Wheel'],
    'Vehicle.Sedan.Driver-Sedan1.Driver':   ['Vehicle', 'Sedan', 'Sedan 1', 'Driver'],
    'Vehicle.Sedan.Body-Sedan2.Body':       ['Vehicle', 'Sedan', 'Sedan 2', 'Body'],
    'Vehicle.Sedan.Door-Sedan2.Door.FL':    ['Vehicle', 'Sedan', 'Sedan 2', 'FL Door'],
    'Vehicle.Sedan.Door-Sedan2.Door.FR':    ['Vehicle', 'Sedan', 'Sedan 2', 'FR Door'],
    'Vehicle.Sedan.Door-Sedan2.Door.BL':    ['Vehicle', 'Sedan', 'Sedan 2', 'BL Door'],
    'Vehicle.Sedan.Door-Sedan2.Door.BR':    ['Vehicle', 'Sedan', 'Sedan 2', 'BR Door'],
    'Vehicle.Sedan.Wheel-Sedan2.Wheel.FL':  ['Vehicle', 'Sedan', 'Sedan 2', 'FL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan2.Wheel.FR':  ['Vehicle', 'Sedan', 'Sedan 2', 'FR Wheel'],
    'Vehicle.Sedan.Wheel-Sedan2.Wheel.BL':  ['Vehicle', 'Sedan', 'Sedan 2', 'BL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan2.Wheel.BR':  ['Vehicle', 'Sedan', 'Sedan 2', 'BR Wheel'],
    'Vehicle.Sedan.Driver-Sedan2.Driver':   ['Vehicle', 'Sedan', 'Sedan 2', 'Driver'],
    'Vehicle.Sedan.Body-Sedan3.Body':       ['Vehicle', 'Sedan', 'Sedan 3', 'Body'],
    'Vehicle.Sedan.Door-Sedan3.Door.FL':    ['Vehicle', 'Sedan', 'Sedan 3', 'FL Door'],
    'Vehicle.Sedan.Door-Sedan3.Door.FR':    ['Vehicle', 'Sedan', 'Sedan 3', 'FR Door'],
    'Vehicle.Sedan.Door-Sedan3.Door.BL':    ['Vehicle', 'Sedan', 'Sedan 3', 'BL Door'],
    'Vehicle.Sedan.Door-Sedan3.Door.BR':    ['Vehicle', 'Sedan', 'Sedan 3', 'BR Door'],
    'Vehicle.Sedan.Wheel-Sedan3.Wheel.FL':  ['Vehicle', 'Sedan', 'Sedan 3', 'FL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan3.Wheel.FR':  ['Vehicle', 'Sedan', 'Sedan 3', 'FR Wheel'],
    'Vehicle.Sedan.Wheel-Sedan3.Wheel.BL':  ['Vehicle', 'Sedan', 'Sedan 3', 'BL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan3.Wheel.BR':  ['Vehicle', 'Sedan', 'Sedan 3', 'BR Wheel'],
    'Vehicle.Sedan.Driver-Sedan3.Driver':   ['Vehicle', 'Sedan', 'Sedan 3', 'Driver'],
    'Vehicle.Sedan.Body-Sedan4.Body':       ['Vehicle', 'Sedan', 'Sedan 4', 'Body'],
    'Vehicle.Sedan.Door-Sedan4.Door.FL':    ['Vehicle', 'Sedan', 'Sedan 4', 'FL Door'],
    'Vehicle.Sedan.Door-Sedan4.Door.FR':    ['Vehicle', 'Sedan', 'Sedan 4', 'FR Door'],
    'Vehicle.Sedan.Door-Sedan4.Door.BL':    ['Vehicle', 'Sedan', 'Sedan 4', 'BL Door'],
    'Vehicle.Sedan.Door-Sedan4.Door.BR':    ['Vehicle', 'Sedan', 'Sedan 4', 'BR Door'],
    'Vehicle.Sedan.Wheel-Sedan4.Wheel.FL':  ['Vehicle', 'Sedan', 'Sedan 4', 'FL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan4.Wheel.FR':  ['Vehicle', 'Sedan', 'Sedan 4', 'FR Wheel'],
    'Vehicle.Sedan.Wheel-Sedan4.Wheel.BL':  ['Vehicle', 'Sedan', 'Sedan 4', 'BL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan4.Wheel.BR':  ['Vehicle', 'Sedan', 'Sedan 4', 'BR Wheel'],
    'Vehicle.Sedan.Driver-Sedan4.Driver':   ['Vehicle', 'Sedan', 'Sedan 4', 'Driver'],
    'Vehicle.Sedan.Body-Sedan5.Body':       ['Vehicle', 'Sedan', 'Sedan 5', 'Body'],
    'Vehicle.Sedan.Door-Sedan5.Door.FL':    ['Vehicle', 'Sedan', 'Sedan 5', 'FL Door'],
    'Vehicle.Sedan.Door-Sedan5.Door.FR':    ['Vehicle', 'Sedan', 'Sedan 5', 'FR Door'],
    'Vehicle.Sedan.Door-Sedan5.Door.BL':    ['Vehicle', 'Sedan', 'Sedan 5', 'BL Door'],
    'Vehicle.Sedan.Door-Sedan5.Door.BR':    ['Vehicle', 'Sedan', 'Sedan 5', 'BR Door'],
    'Vehicle.Sedan.Wheel-Sedan5.Wheel.FL':  ['Vehicle', 'Sedan', 'Sedan 5', 'FL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan5.Wheel.FR':  ['Vehicle', 'Sedan', 'Sedan 5', 'FR Wheel'],
    'Vehicle.Sedan.Wheel-Sedan5.Wheel.BL':  ['Vehicle', 'Sedan', 'Sedan 5', 'BL Wheel'],
    'Vehicle.Sedan.Wheel-Sedan5.Wheel.BR':  ['Vehicle', 'Sedan', 'Sedan 5', 'BR Wheel'],
    'Vehicle.Sedan.Driver-Sedan5.Driver':   ['Vehicle', 'Sedan', 'Sedan 5', 'Driver'],
    
    'Vehicle.SportCar.Body-SportCar1.Body':       ['Vehicle', 'Sport Car', 'Sport Car 1', 'Body'],
    'Vehicle.SportCar.Door-SportCar1.Door.FR':    ['Vehicle', 'Sport Car', 'Sport Car 1', 'FR Door'],
    'Vehicle.SportCar.Door-SportCar1.Door.BL':    ['Vehicle', 'Sport Car', 'Sport Car 1', 'BL Door'],
    'Vehicle.SportCar.Door-SportCar1.Door.FL':    ['Vehicle', 'Sport Car', 'Sport Car 1', 'FL Door'],
    'Vehicle.SportCar.Door-SportCar1.Door.BR':    ['Vehicle', 'Sport Car', 'Sport Car 1', 'BR Door'],
    'Vehicle.SportCar.Wheel-SportCar1.Wheel.FL':  ['Vehicle', 'Sport Car', 'Sport Car 1', 'FL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar1.Wheel.FR':  ['Vehicle', 'Sport Car', 'Sport Car 1', 'FR Wheel'],
    'Vehicle.SportCar.Wheel-SportCar1.Wheel.BL':  ['Vehicle', 'Sport Car', 'Sport Car 1', 'BL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar1.Wheel.BR':  ['Vehicle', 'Sport Car', 'Sport Car 1', 'BR Wheel'],
    'Vehicle.SportCar.Driver-SportCar1.Driver':   ['Vehicle', 'Sport Car', 'Sport Car 1', 'Driver'],
    'Vehicle.SportCar.Body-SportCar2.Body':       ['Vehicle', 'Sport Car', 'Sport Car 2', 'Body'],
    'Vehicle.SportCar.Door-SportCar2.Door.FR':    ['Vehicle', 'Sport Car', 'Sport Car 2', 'FR Door'],
    'Vehicle.SportCar.Door-SportCar2.Door.BL':    ['Vehicle', 'Sport Car', 'Sport Car 2', 'BL Door'],
    'Vehicle.SportCar.Door-SportCar2.Door.FL':    ['Vehicle', 'Sport Car', 'Sport Car 2', 'FL Door'],
    'Vehicle.SportCar.Door-SportCar2.Door.BR':    ['Vehicle', 'Sport Car', 'Sport Car 2', 'BR Door'],
    'Vehicle.SportCar.Wheel-SportCar2.Wheel.FL':  ['Vehicle', 'Sport Car', 'Sport Car 2', 'FL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar2.Wheel.FR':  ['Vehicle', 'Sport Car', 'Sport Car 2', 'FR Wheel'],
    'Vehicle.SportCar.Wheel-SportCar2.Wheel.BL':  ['Vehicle', 'Sport Car', 'Sport Car 2', 'BL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar2.Wheel.BR':  ['Vehicle', 'Sport Car', 'Sport Car 2', 'BR Wheel'],
    'Vehicle.SportCar.Driver-SportCar2.Driver':   ['Vehicle', 'Sport Car', 'Sport Car 2', 'Driver'],
    'Vehicle.SportCar.Body-SportCar3.Body':       ['Vehicle', 'Sport Car', 'Sport Car 3', 'Body'],
    'Vehicle.SportCar.Door-SportCar3.Door.FR':    ['Vehicle', 'Sport Car', 'Sport Car 3', 'FR Door'],
    'Vehicle.SportCar.Door-SportCar3.Door.BL':    ['Vehicle', 'Sport Car', 'Sport Car 3', 'BL Door'],
    'Vehicle.SportCar.Door-SportCar3.Door.FL':    ['Vehicle', 'Sport Car', 'Sport Car 3', 'FL Door'],
    'Vehicle.SportCar.Door-SportCar3.Door.BR':    ['Vehicle', 'Sport Car', 'Sport Car 3', 'BR Door'],
    'Vehicle.SportCar.Wheel-SportCar3.Wheel.FL':  ['Vehicle', 'Sport Car', 'Sport Car 3', 'FL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar3.Wheel.FR':  ['Vehicle', 'Sport Car', 'Sport Car 3', 'FR Wheel'],
    'Vehicle.SportCar.Wheel-SportCar3.Wheel.BL':  ['Vehicle', 'Sport Car', 'Sport Car 3', 'BL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar3.Wheel.BR':  ['Vehicle', 'Sport Car', 'Sport Car 3', 'BR Wheel'],
    'Vehicle.SportCar.Driver-SportCar3.Driver':   ['Vehicle', 'Sport Car', 'Sport Car 3', 'Driver'],
    'Vehicle.SportCar.Body-SportCar4.Body':       ['Vehicle', 'Sport Car', 'Sport Car 4', 'Body'],
    'Vehicle.SportCar.Door-SportCar4.Door.FR':    ['Vehicle', 'Sport Car', 'Sport Car 4', 'FR Door'],
    'Vehicle.SportCar.Door-SportCar4.Door.BL':    ['Vehicle', 'Sport Car', 'Sport Car 4', 'BL Door'],
    'Vehicle.SportCar.Door-SportCar4.Door.FL':    ['Vehicle', 'Sport Car', 'Sport Car 4', 'FL Door'],
    'Vehicle.SportCar.Door-SportCar4.Door.BR':    ['Vehicle', 'Sport Car', 'Sport Car 4', 'BR Door'],
    'Vehicle.SportCar.Wheel-SportCar4.Wheel.FL':  ['Vehicle', 'Sport Car', 'Sport Car 4', 'FL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar4.Wheel.FR':  ['Vehicle', 'Sport Car', 'Sport Car 4', 'FR Wheel'],
    'Vehicle.SportCar.Wheel-SportCar4.Wheel.BL':  ['Vehicle', 'Sport Car', 'Sport Car 4', 'BL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar4.Wheel.BR':  ['Vehicle', 'Sport Car', 'Sport Car 4', 'BR Wheel'],
    'Vehicle.SportCar.Driver-SportCar4.Driver':   ['Vehicle', 'Sport Car', 'Sport Car 4', 'Driver'],
    'Vehicle.SportCar.Body-SportCar5.Body':       ['Vehicle', 'Sport Car', 'Sport Car 5', 'Body'],
    'Vehicle.SportCar.Door-SportCar5.Door.FR':    ['Vehicle', 'Sport Car', 'Sport Car 5', 'FR Door'],
    'Vehicle.SportCar.Door-SportCar5.Door.BL':    ['Vehicle', 'Sport Car', 'Sport Car 5', 'BL Door'],
    'Vehicle.SportCar.Door-SportCar5.Door.FL':    ['Vehicle', 'Sport Car', 'Sport Car 5', 'FL Door'],
    'Vehicle.SportCar.Door-SportCar5.Door.BR':    ['Vehicle', 'Sport Car', 'Sport Car 5', 'BR Door'],
    'Vehicle.SportCar.Wheel-SportCar5.Wheel.FL':  ['Vehicle', 'Sport Car', 'Sport Car 5', 'FL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar5.Wheel.FR':  ['Vehicle', 'Sport Car', 'Sport Car 5', 'FR Wheel'],
    'Vehicle.SportCar.Wheel-SportCar5.Wheel.BL':  ['Vehicle', 'Sport Car', 'Sport Car 5', 'BL Wheel'],
    'Vehicle.SportCar.Wheel-SportCar5.Wheel.BR':  ['Vehicle', 'Sport Car', 'Sport Car 5', 'BR Wheel'],
    'Vehicle.SportCar.Driver-SportCar5.Driver':   ['Vehicle', 'Sport Car', 'Sport Car 5', 'Driver'],

    'Vehicle.Jeep.Body-Jeep1.Body':       ['Vehicle', 'Jeep', 'Jeep 1', 'Body'],
    'Vehicle.Jeep.Door-Jeep1.Door.FR':    ['Vehicle', 'Jeep', 'Jeep 1', 'FR Door'],
    'Vehicle.Jeep.Door-Jeep1.Door.BL':    ['Vehicle', 'Jeep', 'Jeep 1', 'BL Door'],
    'Vehicle.Jeep.Door-Jeep1.Door.FL':    ['Vehicle', 'Jeep', 'Jeep 1', 'FL Door'],
    'Vehicle.Jeep.Door-Jeep1.Door.BR':    ['Vehicle', 'Jeep', 'Jeep 1', 'BR Door'],
    'Vehicle.Jeep.Wheel-Jeep1.Wheel.FL':  ['Vehicle', 'Jeep', 'Jeep 1', 'FL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep1.Wheel.FR':  ['Vehicle', 'Jeep', 'Jeep 1', 'FR Wheel'],
    'Vehicle.Jeep.Wheel-Jeep1.Wheel.BL':  ['Vehicle', 'Jeep', 'Jeep 1', 'BL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep1.Wheel.BR':  ['Vehicle', 'Jeep', 'Jeep 1', 'BR Wheel'],
    'Vehicle.Jeep.Driver-Jeep1.Driver':   ['Vehicle', 'Jeep', 'Jeep 1', 'Driver'],
    'Vehicle.Jeep.Body-Jeep2.Body':       ['Vehicle', 'Jeep', 'Jeep 2', 'Body'],
    'Vehicle.Jeep.Door-Jeep2.Door.FR':    ['Vehicle', 'Jeep', 'Jeep 2', 'FR Door'],
    'Vehicle.Jeep.Door-Jeep2.Door.BL':    ['Vehicle', 'Jeep', 'Jeep 2', 'BL Door'],
    'Vehicle.Jeep.Door-Jeep2.Door.FL':    ['Vehicle', 'Jeep', 'Jeep 2', 'FL Door'],
    'Vehicle.Jeep.Door-Jeep2.Door.BR':    ['Vehicle', 'Jeep', 'Jeep 2', 'BR Door'],
    'Vehicle.Jeep.Wheel-Jeep2.Wheel.FL':  ['Vehicle', 'Jeep', 'Jeep 2', 'FL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep2.Wheel.FR':  ['Vehicle', 'Jeep', 'Jeep 2', 'FR Wheel'],
    'Vehicle.Jeep.Wheel-Jeep2.Wheel.BL':  ['Vehicle', 'Jeep', 'Jeep 2', 'BL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep2.Wheel.BR':  ['Vehicle', 'Jeep', 'Jeep 2', 'BR Wheel'],
    'Vehicle.Jeep.Driver-Jeep2.Driver':   ['Vehicle', 'Jeep', 'Jeep 2', 'Driver'],
    'Vehicle.Jeep.Body-Jeep3.Body':       ['Vehicle', 'Jeep', 'Jeep 3', 'Body'],
    'Vehicle.Jeep.Door-Jeep3.Door.FR':    ['Vehicle', 'Jeep', 'Jeep 3', 'FR Door'],
    'Vehicle.Jeep.Door-Jeep3.Door.BL':    ['Vehicle', 'Jeep', 'Jeep 3', 'BL Door'],
    'Vehicle.Jeep.Door-Jeep3.Door.FL':    ['Vehicle', 'Jeep', 'Jeep 3', 'FL Door'],
    'Vehicle.Jeep.Door-Jeep3.Door.BR':    ['Vehicle', 'Jeep', 'Jeep 3', 'BR Door'],
    'Vehicle.Jeep.Wheel-Jeep3.Wheel.FL':  ['Vehicle', 'Jeep', 'Jeep 3', 'FL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep3.Wheel.FR':  ['Vehicle', 'Jeep', 'Jeep 3', 'FR Wheel'],
    'Vehicle.Jeep.Wheel-Jeep3.Wheel.BL':  ['Vehicle', 'Jeep', 'Jeep 3', 'BL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep3.Wheel.BR':  ['Vehicle', 'Jeep', 'Jeep 3', 'BR Wheel'],
    'Vehicle.Jeep.Driver-Jeep3.Driver':   ['Vehicle', 'Jeep', 'Jeep 3', 'Driver'],
    'Vehicle.Jeep.Body-Jeep4.Body':       ['Vehicle', 'Jeep', 'Jeep 4', 'Body'],
    'Vehicle.Jeep.Door-Jeep4.Door.FR':    ['Vehicle', 'Jeep', 'Jeep 4', 'FR Door'],
    'Vehicle.Jeep.Door-Jeep4.Door.BL':    ['Vehicle', 'Jeep', 'Jeep 4', 'BL Door'],
    'Vehicle.Jeep.Door-Jeep4.Door.FL':    ['Vehicle', 'Jeep', 'Jeep 4', 'FL Door'],
    'Vehicle.Jeep.Door-Jeep4.Door.BR':    ['Vehicle', 'Jeep', 'Jeep 4', 'BR Door'],
    'Vehicle.Jeep.Wheel-Jeep4.Wheel.FL':  ['Vehicle', 'Jeep', 'Jeep 4', 'FL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep4.Wheel.FR':  ['Vehicle', 'Jeep', 'Jeep 4', 'FR Wheel'],
    'Vehicle.Jeep.Wheel-Jeep4.Wheel.BL':  ['Vehicle', 'Jeep', 'Jeep 4', 'BL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep4.Wheel.BR':  ['Vehicle', 'Jeep', 'Jeep 4', 'BR Wheel'],
    'Vehicle.Jeep.Driver-Jeep4.Driver':   ['Vehicle', 'Jeep', 'Jeep 4', 'Driver'],
    'Vehicle.Jeep.Body-Jeep5.Body':       ['Vehicle', 'Jeep', 'Jeep 5', 'Body'],
    'Vehicle.Jeep.Door-Jeep5.Door.FR':    ['Vehicle', 'Jeep', 'Jeep 5', 'FR Door'],
    'Vehicle.Jeep.Door-Jeep5.Door.BL':    ['Vehicle', 'Jeep', 'Jeep 5', 'BL Door'],
    'Vehicle.Jeep.Door-Jeep5.Door.FL':    ['Vehicle', 'Jeep', 'Jeep 5', 'FL Door'],
    'Vehicle.Jeep.Door-Jeep5.Door.BR':    ['Vehicle', 'Jeep', 'Jeep 5', 'BR Door'],
    'Vehicle.Jeep.Wheel-Jeep5.Wheel.FL':  ['Vehicle', 'Jeep', 'Jeep 5', 'FL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep5.Wheel.FR':  ['Vehicle', 'Jeep', 'Jeep 5', 'FR Wheel'],
    'Vehicle.Jeep.Wheel-Jeep5.Wheel.BL':  ['Vehicle', 'Jeep', 'Jeep 5', 'BL Wheel'],
    'Vehicle.Jeep.Wheel-Jeep5.Wheel.BR':  ['Vehicle', 'Jeep', 'Jeep 5', 'BR Wheel'],
    'Vehicle.Jeep.Driver-Jeep5.Driver':   ['Vehicle', 'Jeep', 'Jeep 5', 'Driver'],

    'Vehicle.MicroBus.Body-MicroBus1.Body':       ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'Body'],
    'Vehicle.MicroBus.Door-MicroBus1.Door.FR':    ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'FR Door'],
    'Vehicle.MicroBus.Door-MicroBus1.Door.BL':    ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'BL Door'],
    'Vehicle.MicroBus.Door-MicroBus1.Door.FL':    ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'FL Door'],
    'Vehicle.MicroBus.Door-MicroBus1.Door.BR':    ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'BR Door'],
    'Vehicle.MicroBus.Wheel-MicroBus1.Wheel.FL':  ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'FL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus1.Wheel.FR':  ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'FR Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus1.Wheel.BL':  ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'BL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus1.Wheel.BR':  ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'BR Wheel'],
    'Vehicle.MicroBus.Driver-MicroBus1.Driver':   ['Vehicle', 'Micro Bus', 'Micro Bus 1', 'Driver'],
    'Vehicle.MicroBus.Body-MicroBus2.Body':       ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'Body'],
    'Vehicle.MicroBus.Door-MicroBus2.Door.FR':    ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FR Door'],
    'Vehicle.MicroBus.Door-MicroBus2.Door.BL':    ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BL Door'],
    'Vehicle.MicroBus.Door-MicroBus2.Door.FL':    ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FL Door'],
    'Vehicle.MicroBus.Door-MicroBus2.Door.BR':    ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BR Door'],
    'Vehicle.MicroBus.Door-MicroBus2.Door.Back':  ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'Back Door'],
    'Vehicle.MicroBus.Wheel-MicroBus2.Wheel.FL':  ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FL Wheel'],
    'Vehicle.MicroBus2.Wheel-MicroBus2.Wheel.FL': ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus2.Wheel.FR':  ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FR Wheel'],
    'Vehicle.MicroBus2.Wheel-MicroBus2.Wheel.FR': ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'FR Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus2.Wheel.BL':  ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BL Wheel'],
    'Vehicle.MicroBus2.Wheel-MicroBus2.Wheel.BL': ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus2.Wheel.BR':  ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BR Wheel'],
    'Vehicle.MicroBus2.Wheel-MicroBus2.Wheel.BR': ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'BR Wheel'],
    'Vehicle.MicroBus.Driver-MicroBus2.Driver':   ['Vehicle', 'Micro Bus', 'Micro Bus 2', 'Driver'],
    'Vehicle.MicroBus.Body-MicroBus3.Body':       ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'Body'],
    'Vehicle.MicroBus.Door-MicroBus3.Door.FR':    ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'FR Door'],
    'Vehicle.MicroBus.Door-MicroBus3.Door.BL':    ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'BL Door'],
    'Vehicle.MicroBus.Door-MicroBus3.Door.FL':    ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'FL Door'],
    'Vehicle.MicroBus.Door-MicroBus3.Door.BR':    ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'BR Door'],
    'Vehicle.MicroBus.Wheel-MicroBus3.Wheel.FL':  ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'FL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus3.Wheel.FR':  ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'FR Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus3.Wheel.BL':  ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'BL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus3.Wheel.BR':  ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'BR Wheel'],
    'Vehicle.MicroBus.Driver-MicroBus3.Driver':   ['Vehicle', 'Micro Bus', 'Micro Bus 3', 'Driver'],
    'Vehicle.MicroBus.Body-MicroBus4.Body':       ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'Body'],
    'Vehicle.MicroBus.Door-MicroBus4.Door.FR':    ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'FR Door'],
    'Vehicle.MicroBus.Door-MicroBus4.Door.BL':    ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'BL Door'],
    'Vehicle.MicroBus.Door-MicroBus4.Door.FL':    ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'FL Door'],
    'Vehicle.MicroBus.Door-MicroBus4.Door.BR':    ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'BR Door'],
    'Vehicle.MicroBus.Wheel-MicroBus4.Wheel.FL':  ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'FL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus4.Wheel.FR':  ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'FR Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus4.Wheel.BL':  ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'BL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus4.Wheel.BR':  ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'BR Wheel'],
    'Vehicle.MicroBus.Driver-MicroBus4.Driver':   ['Vehicle', 'Micro Bus', 'Micro Bus 4', 'Driver'],
    'Vehicle.MicroBus.Body-MicroBus5.Body':       ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'Body'],
    'Vehicle.MicroBus.Door-MicroBus5.Door.FR':    ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'FR Door'],
    'Vehicle.MicroBus.Door-MicroBus5.Door.BL':    ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'BL Door'],
    'Vehicle.MicroBus.Door-MicroBus5.Door.FL':    ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'FL Door'],
    'Vehicle.MicroBus.Door-MicroBus5.Door.BR':    ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'BR Door'],
    'Vehicle.MicroBus.Wheel-MicroBus5.Wheel.FL':  ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'FL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus5.Wheel.FR':  ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'FR Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus5.Wheel.BL':  ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'BL Wheel'],
    'Vehicle.MicroBus.Wheel-MicroBus5.Wheel.BR':  ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'BR Wheel'],
    'Vehicle.MicroBus.Driver-MicroBus5.Driver':   ['Vehicle', 'Micro Bus', 'Micro Bus 5', 'Driver'],

    'Vehicle.Truck.Body-Truck1.Body':       ['Vehicle', 'Truck', 'Truck 1', 'Body'],
    'Vehicle.Truck.Door-Truck1.Door.FL':    ['Vehicle', 'Truck', 'Truck 1', 'FL Door'], 
    'Vehicle.Truck.Door-Truck1.Door.FR':    ['Vehicle', 'Truck', 'Truck 1', 'FR Door'],
    'Vehicle.Truck.Wheel-Truck1.Wheel.FR':  ['Vehicle', 'Truck', 'Truck 1', 'FR Wheel'],
    'Vehicle.Truck.Wheel-Truck1.Wheel.FL':  ['Vehicle', 'Truck', 'Truck 1', 'FL Wheel'],
    'Vehicle.Truck.Wheel-Truck1.Wheel.BR':  ['Vehicle', 'Truck', 'Truck 1', 'BR Wheel'], 
    'Vehicle.Truck.Wheel-Truck1.Wheel.BL':  ['Vehicle', 'Truck', 'Truck 1', 'BL Wheel'],
    'Vehicle.Truck.Wheel-Truck1.Wheel.BL2': ['Vehicle', 'Truck', 'Truck 1', 'BL2 Wheel'],
    'Vehicle.Truck.Wheel-Truck1.Wheel.BR2': ['Vehicle', 'Truck', 'Truck 1', 'BR2 Wheel'],
    'Vehicle.Truck.Driver-Truck1.Driver':   ['Vehicle', 'Truck', 'Truck 1', 'Driver'],
}



# =================================
# Normalization
# =================================

def normalize(vx, vy, vz):
    mag = np.sqrt(vx**2 + vy**2 + vz**2)
    # Replace zeros with 1 to avoid division issues
    mag_safe = np.where(mag == 0, 1, mag)
    return vx/mag_safe, vy/mag_safe, vz/mag_safe

# =================================
# Calculating angular velocity
# =================================

def angular_velocity(vx, vy, vz, dt):
    dot = (
        vx * vx.shift(1) +
        vy * vy.shift(1) +
        vz * vz.shift(1)
    )
    # HARD clamp (important)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)
    ang_vel = angle / dt
    return ang_vel.fillna(0)

# =================================
# Calculating velocity direction, with normalization
# =================================

def velocity_direction(vx, vy, vz):
    dvx = vx.diff()
    dvy = vy.diff()
    dvz = vz.diff()
    mag = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    # Avoid division by zero AND NaNs
    mag = mag.replace(0, np.nan)
    vdx = dvx / mag
    vdy = dvy / mag
    vdz = dvz / mag
    return vdx, vdy, vdz

# =================================
# Helper for target gaze consistency
# =================================

def rolling_mode_fraction_fast(arr, window):
    out = np.zeros(len(arr))

    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_vals = arr[start:i+1]

        if len(window_vals) == 0:
            continue

        counts = Counter(window_vals)
        mode_count = counts.most_common(1)[0][1]
        out[i] = mode_count / len(window_vals)

    return out


# =================================
# Extracting eye motion from eye data
# =================================

def extract_eye_motion(
    eye_df:pd.DataFrame,
    WINDOW = 5,  # adjust (~50–100 ms depending on framerate)
    EYE_LOW = 0.5,
    HEAD_LOW = 0.5,
    HEAD_HIGH = 1.5,
    ALIGN_VOR = -0.5,
    ALIGN_PURSUIT = 0.5,
    DRIFT_LOW = 0.1,
    DRIFT_HIGH = 0.5,
    CONSISTENCY_MED = 0.5,
    CONSISTENCY_HIGH = 0.8,
    MIN_FRAMES = 3
):
    required_cols = [
        "head_direction_x",
        "head_direction_y",
        "head_direction_z",
        "gaze_head_rel_direction_x",
        "gaze_head_rel_direction_y",
        "gaze_head_rel_direction_z",
        "unix_ms"
    ]

     # Copy to prevent mutation
    df = eye_df.copy() 
    # Drop the last row just to be safe, as some files are incomplete
    df = df.iloc[:-1]
    
    # 1. Caclulate delta time
    df["dt"] = df["unix_ms"].diff() / 1000.0    # Extract delta time
    df["dt"] = df["dt"].fillna(method="bfill").clip(lower=1e-6)

    # Factorize (generate integer-based labels) for `gaze_target_name`
    df["gaze_target_id"], uniques = pd.factorize(df["gaze_target_name"])

    # 2. Normalize Direction Vectors
    # Head direction (unit)
    df["hdx"], df["hdy"], df["hdz"] = normalize(
        df["head_direction_x"],
        df["head_direction_y"],
        df["head_direction_z"]
    )
    # Eye relative to head (unit)
    df["edx"], df["edy"], df["edz"] = normalize(
        df["gaze_head_rel_direction_x"],
        df["gaze_head_rel_direction_y"],
        df["gaze_head_rel_direction_z"]
    )

    # 3. Angular Velocity
    # Eye-in-head angular velocity
    df["eye_ang_vel"] = angular_velocity(
        df["edx"], df["edy"], df["edz"], df["dt"]
    )
    df["eye_ang_vel"] = df["eye_ang_vel"].fillna(0)
    # Head angular velocity
    df["head_ang_vel"] = angular_velocity(
        df["hdx"], df["hdy"], df["hdz"], df["dt"]
    )
    df["head_ang_vel"] = df["head_ang_vel"].fillna(0)

    # 4. Alignment
    # Direction of motion
    df["eye_vdx"], df["eye_vdy"], df["eye_vdz"] = velocity_direction(
        df["edx"], df["edy"], df["edz"]
    )
    df["head_vdx"], df["head_vdy"], df["head_vdz"] = velocity_direction(
        df["hdx"], df["hdy"], df["hdz"]
    )
    df[[
        "eye_vdx", "eye_vdy", "eye_vdz",
        "head_vdx", "head_vdy", "head_vdz"
    ]] = df[[
        "eye_vdx", "eye_vdy", "eye_vdz",
        "head_vdx", "head_vdy", "head_vdz"
    ]].fillna(0)
    # Alignment
    df["alignment"] = (
        df["eye_vdx"] * df["head_vdx"] +
        df["eye_vdy"] * df["head_vdy"] +
        df["eye_vdz"] * df["head_vdz"]
    ).clip(-1, 1)
    df["alignment"] = df["alignment"].fillna(0)

    # 5. Windows Target Alignment
    df["target_consistency"] = rolling_mode_fraction_fast(
        df["gaze_target_id"].values, WINDOW
    )

    # 6. Rolling Features
    df["eye_vel_mean"] = df["eye_ang_vel"].rolling(WINDOW).mean()
    df["head_vel_mean"] = df["head_ang_vel"].rolling(WINDOW).mean()
    df["alignment_mean"] = df["alignment"].rolling(WINDOW).mean()
    df["angle_diff_std"] = df["gaze_head_angle_diff"].rolling(WINDOW).std()

    # 7. Classification
    conditions = [
        # Fixation
        (
            (df["eye_vel_mean"] < EYE_LOW) &
            (df["head_vel_mean"] < HEAD_LOW) &
            (df["target_consistency"] > CONSISTENCY_HIGH)
        ),
        # Drift
        (
            (df["eye_vel_mean"] > DRIFT_LOW) &
            (df["eye_vel_mean"] < DRIFT_HIGH) &
            (df["head_vel_mean"] < HEAD_LOW) &
            (df["target_consistency"] < CONSISTENCY_HIGH) &
            (df["target_consistency"] > CONSISTENCY_MED)
        ),
        # VOR
        (
            (df["head_vel_mean"] > HEAD_HIGH) &
            (df["eye_vel_mean"] > EYE_LOW) &
            (df["alignment_mean"] < ALIGN_VOR) &
            (df["target_consistency"] > CONSISTENCY_HIGH)
        ),

        # Smooth pursuit
        (
            (df["eye_vel_mean"] > EYE_LOW) &
            (df["alignment_mean"] > ALIGN_PURSUIT) &
            (df["target_consistency"] > CONSISTENCY_HIGH)
        ),
        
    ]
    choices = [
        "fixation",
        "drift",
        "vor",
        "smooth_pursuit"
    ]
    df["gaze_state"] = np.select(conditions, choices, default="general_movement")

    # 8. Remove very short segments (debounce)
    df["state_change"] = df["gaze_state"] != df["gaze_state"].shift()
    df["segment_id"] = df["state_change"].cumsum()
    segment_sizes = df.groupby("segment_id").size()
    small_segments = segment_sizes[segment_sizes < MIN_FRAMES].index
    df.loc[df["segment_id"].isin(small_segments), "gaze_state"] = "general_movement"

    # Finally, return
    return df