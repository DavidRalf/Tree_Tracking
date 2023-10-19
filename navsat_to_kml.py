#!/usr/bin/env python3

import rospy
import random
import simplekml
from sensor_msgs.msg import NavSatFix   

class NavsatToKml:
    def __init__(self):
        self.nav_topic = rospy.get_param('~topic', '/ublox/fix')
        self.filename = rospy.get_param('~file', 'navsat.kml')
        self.nav_sub = rospy.Subscriber(self.nav_topic, NavSatFix, self.nav_cb, queue_size=10)
        print(f"Subscribing to Navsat {self.nav_topic}")
        self.kml = simplekml.Kml()
        self.line = self.kml.newlinestring(name=f"Line {self.nav_topic}")
        self.line.altitudemode = simplekml.AltitudeMode.clamptoground
        self.line.linestyle.width = 5
        self.line.linestyle.color = simplekml.Color.blue
        self.coords = []

    def nav_cb(self, msg: NavSatFix):
        self.coords.append((msg.longitude, msg.latitude, msg.altitude))
        print(f"{len(self.coords)} Points", end="\r", flush=True)
        
    def save_file(self):
        print(f"\nSaving to {self.filename}")
        self.line.coords = self.coords
        self.kml.save(self.filename)

if __name__ == '__main__':
    rospy.init_node(f'navsat_to_kml_{random.randint(0,10000)}')
    i = NavsatToKml()
    rospy.spin()
    i.save_file()