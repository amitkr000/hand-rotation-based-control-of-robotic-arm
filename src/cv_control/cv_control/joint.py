#! /usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class JointManager(Node):
	
	def __init__(self):
		super().__init__("joint_manager")
		
		self.subscription = self.create_subscription(Image, 'video_frames', self.listener_callback, 10)
		
		self.publisher_ = self.create_publisher(Int64,'Hand_Pose', 10)
		
		# We will publish a message every 0.1 seconds
		timer_period = 0.1
		
		# initialize mediapipe
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
		# Load the gesture recognizer model
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model = load_model(os.path.join(dir_path, "handPose"))
		self.get_logger().info('Model_loded')
		
		# Create the timer
		self.timer = self.create_timer(timer_period, self.timer_callback)
		
		# Convert the image data of cv2 into ros message and vice-versa also.
		self.br = CvBridge()
		
		# It store the current frame of the live video
		self.current_frame = np.zeros((2,3))  #dummy zero array
		
		# Convert the prediction of handpose from numpy64 into stdmsg.msg.int64
		self.prediction_msg = int64()
		
	def listener_callback(self, data):
	
		#self.get_logger().info('Receiving video frame')
	
		self.current_frame = self.br.imgmsg_to_cv2(data)
		
	def timer_callback(self):
		self.get_logger().info('Publishing handPose')
		#frame = self.br.imgmsg_to_cv2(Imagedata)
		frame = self.current_frame
		#print(frame)
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
				prediction = self.model.predict([landmarks])
				classID = np.argmax(prediction)
				if classID != None:
					self.prediction_msg.data = int(classID)
					self.publisher_.publish(self.prediction_msg)
					print(classID)
				
def main(args=None):

	rclpy.init(args=args)
	
	joint = JointManager()

	rclpy.spin(joint)

	image_sub.destroy_node()
	
	rclpy.shotdown()
	
if __name__ == '__main__':
	main()
