#include "tensorrt_faster_rcnn_ros/trt_node.h"

extern void setup(std::string planFilename, float nms_th, float score_th);
extern void destroy(void);
extern object_msgs::ObjectsInBoxes infer(const sensor_msgs::ImageConstPtr& color_msg);

ObjectDetection::ObjectDetection(const ros::NodeHandle& nh) : nh_(nh), it_(nh), params()
{
  setup(params.plan_filename, params.nms_threshold, params.score_threshold);
  color_sub = it_.subscribe("/camera/color/image_raw", 10, &ObjectDetection::colorCallback, this);
  bboxes_pub = nh_.advertise<object_msgs::ObjectsInBoxes>("/faster_rcnn_infer/bounding_boxes", 10);
}

ObjectDetection::~ObjectDetection()
{
  destroy();
}

void ObjectDetection::colorCallback(const sensor_msgs::ImageConstPtr& color_msg)
{
  bboxes_msg = infer(color_msg);
  bboxes_pub.publish(bboxes_msg);
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "faster_rcnn_infer");

  ros::NodeHandle nh("~");
  ObjectDetection faster_rcnn(nh);

  ros::spin();

  return 0;
}