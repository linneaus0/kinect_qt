#!/usr/bin/env python
import os
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Receiver:
    def __init__(self):
        self.dir = './recv'
        self.count = 0
        self.bridge = CvBridge()

    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        self.count += 1
        window_name = 'image'
        # 每隔一秒保存一张图片看一下效果
        if self.count % 1 == 0:
            #cv2.imwrite(os.path.join(self.dir, '%s.jpg' % self.count), image)
            cv2.imshow(window_name,image)
            cv2.waitKey(1)
    def receive(self):
        rospy.init_node('receiver', anonymous=True)
        rospy.Subscriber('skeleton_frame', Image, self.callback)
        rospy.spin()

def main():
    receiver = Receiver()
    receiver.receive()

if __name__ == '__main__':
    main()
