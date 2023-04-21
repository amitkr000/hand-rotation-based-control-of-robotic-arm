#! /usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class ImageSubscriber(Node):
	
	def __init__(self):
		super().__init__("handpose_sub")
		
		self.subscription = self.create_subscription(Image, 'video_frames', self.listener_callback, 10)
		
		self.br = CvBridge()
		self.current_frame = None
		
	def listener_callback(self, data):
	
		self.get_logger().info('Receiving video frame')
	
		self.current_frame = self.br.imgmsg_to_cv2(data)
	
		#self.current_frame = data
		
		
class PosePublisher(Node):
	def __init__(self, Imagedata):
		super().__init__("handpose_pub")
		self.publisher_ = self.create_publisher(Int8,'Hand_Pose', 10)
		# We will publish a message every 0.1 seconds
		timer_period = 0.1
		self.br = CvBridge()
		# initialize mediapipe
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
		# Load the gesture recognizer model
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model = load_model(os.path.join(dir_path, "handPose"))
		self.get_logger().info('Model_loded')
		# Create the timer
		self.timer = self.create_timer(timer_period, self.timer_callback(Imagedata))
		
	def timer_callback(self, Imagedata):
		self.get_logger().info('Publishing handPose')
		#frame = self.br.imgmsg_to_cv2(Imagedata)
		frame = Imagedata
		print(frame)
		if (frame != None):
			framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# Get hand landmark prediction
			result = self.hands.process(framergb)
    	
    	# post process the result
			if result.multi_hand_landmarks:
				landmarks = []
				for handslms in result.multi_hand_landmarks:
					for lm in handslms.landmark:
						landmarks.append(lm.x)
						landmarks.append(lm.y)
					#landmarks.append(lm.z)
        # Predict gesture in Hand Gesture Recognition project
			prediction = self.model.predict([landmarks])
			classID = np.argmax(prediction)
			if classID != None:
				self.publisher_.publish(classID)
        	# Display the message on the console
			#self.get_logger().info('Publishing handPose')
		
		
		
def main(args=None):

	rclpy.init(args=args)
	
	
	
	image_sub = ImageSubscriber()
	handPose_pub = PosePublisher(image_sub.current_frame)
	rclpy.spin(image_sub,handPose_pub)
	rclpy.spin(handPose_pub)
	
	
	
	
	image_sub.destroy_node()
	handPose_pub.destroy_node()
	
	rclpy.shotdown()
	
if __name__ == '__main__':
	main()
