#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

class ImagePublisher(Node):
	
	def __init__(self):
		super().__init__("image_pub")
		# Create the publisher. This publisher will publish an Image
    	# to the video_frames topic. The queue size is 10 messages.
		self.publisher_ = self.create_publisher(Image,'video_frames', 10)
		# We will publish a message every 0.1 seconds
		timer_period = 0.1
		
		# Create the timer
		self.timer = self.create_timer(timer_period, self.timer_callback)
		
		self.cap = cv2.VideoCapture(0)
		
		self.br = CvBridge()
		
	def timer_callback(self):
		#Callback function.
    	#This function gets called every 0.1 seconds.
    	# Capture frame-by-frame
    	# This method returns True/False as well
    	# as the video frame.
		ret, frame = self.cap.read()
		
		if ret == True:
		# Publish the image.
      	# The 'cv2_to_imgmsg' method converts an OpenCV
      	# image to a ROS 2 image message
			self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
		
		# Display the message on the console
		self.get_logger().info('Publishing video frame')
    	


def main(args=None):
	rclpy.init(args=args)
	image_pub = ImagePublisher()
	rclpy.spin(image_pub)
	image_publisher.destroy_node()
	rclpy.shutdown()
	
	
if __name__== '__main__':
	main()
