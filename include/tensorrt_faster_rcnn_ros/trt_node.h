#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "object_msgs/ObjectsInBoxes.h"

class ObjectDetection
{
public:
  ObjectDetection(const ros::NodeHandle& nh);
  ~ObjectDetection();

  struct parameters
  {
    std::string plan_filename;
    float nms_threshold;
    float score_threshold;

    parameters()
    {
      ros::param::param<std::string>("/faster_rcnn_infer/plan_filename", plan_filename, "/home/borong/catkin_ws/src/"
                                                                                        "tensorrt_faster_rcnn_ros/data/"
                                                                                        "VGG16_faster_rcnn_final.plan");
      ros::param::param<float>("/faster_rcnn_infer/nms_threshold", nms_threshold, 0.3);
      ros::param::param<float>("/faster_rcnn_infer/score_threshold", score_threshold, 0.8);
    }
  };
  const struct parameters params;

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber color_sub;
  ros::Publisher bboxes_pub;
  object_msgs::ObjectsInBoxes bboxes_msg;

  void colorCallback(const sensor_msgs::ImageConstPtr& msg);
};