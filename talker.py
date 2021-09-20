#!/usr/bin/env python

from os import listdir
from os.path import isfile, join

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Sender:
    def __init__(self, fps=2, dir='/home/ubuntu/Pictures/test', loop=10):
        self.fps = fps
        self.dir = dir
        self.loop = loop

    def send(self):
        rospy.init_node('sender', anonymous=True)
        pub = rospy.Publisher('image_flow', Image, queue_size=1)
        bridge = CvBridge()
        rate = rospy.Rate(self.fps)

        files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        files.sort()
        
        while self.loop != 0:
            self.loop -= 1
            print(self.loop)
            for file in files:
                if not rospy.is_shutdown():
                    image = cv2.imread(join(self.dir, file))
                    msg = bridge.cv2_to_imgmsg(image, 'bgr8')
                    pub.publish(msg)
                    rate.sleep()
                else:
                    return

def main():
    sender = Sender()
    sender.send()

if __name__ == '__main__':
    main()