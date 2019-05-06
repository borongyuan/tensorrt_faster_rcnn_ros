#include <cassert>
#include <fstream>
#include <cmath>
#include <chrono>
#include <memory>
#include <cstring>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "object_msgs/ObjectsInBoxes.h"

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define CHECK(status)                                                                                                  \
  {                                                                                                                    \
    if (status != 0)                                                                                                   \
    {                                                                                                                  \
      std::cout << "Cuda failure: " << status;                                                                         \
      abort();                                                                                                         \
    }                                                                                                                  \
  }

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
const size_t stridesCv[3] = { INPUT_W * INPUT_C, INPUT_C, 1 };
const size_t strides[3] = { INPUT_H * INPUT_W, INPUT_W, 1 };
const float pixelMean[3] = { 102.9801f, 115.9465f, 122.7717f };

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane",   "bicycle", "bird",  "boat",
                                            "bottle",     "bus",         "car",     "cat",   "chair",
                                            "cow",        "diningtable", "dog",     "horse", "motorbike",
                                            "person",     "pottedplant", "sheep",   "sofa",  "train",
                                            "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

float nms_threshold, score_threshold;

class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    if (severity != Severity::kINFO)
      ROS_INFO("[[trt_infer.cu]] %s", msg);
  }
} gLogger;

template <int OutC>
class Reshape : public IPlugin
{
public:
  Reshape()
  {
  }
  Reshape(const void* buffer, size_t size)
  {
    assert(size == sizeof(mCopySize));
    mCopySize = *reinterpret_cast<const size_t*>(buffer);
  }

  int getNbOutputs() const override
  {
    return 1;
  }
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
  {
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);
    assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);
    return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
  }

  int initialize() override
  {
    return 0;
  }

  void terminate() override
  {
  }

  size_t getWorkspaceSize(int) const override
  {
    return 0;
  }

  // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the
  // output buffer
  int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
  {
    CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
    return 0;
  }

  size_t getSerializationSize() override
  {
    return sizeof(mCopySize);
  }

  void serialize(void* buffer) override
  {
    *reinterpret_cast<size_t*>(buffer) = mCopySize;
  }

  void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
  {
    mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
  }

protected:
  size_t mCopySize;
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
  // deserialization plugin implementation
  virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                                          int nbWeights) override
  {
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "ReshapeCTo2"))
    {
      assert(mPluginRshp2 == nullptr);
      assert(nbWeights == 0 && weights == nullptr);
      mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
      return mPluginRshp2.get();
    }
    else if (!strcmp(layerName, "ReshapeCTo18"))
    {
      assert(mPluginRshp18 == nullptr);
      assert(nbWeights == 0 && weights == nullptr);
      mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
      return mPluginRshp18.get();
    }
    else if (!strcmp(layerName, "RPROIFused"))
    {
      assert(mPluginRPROI == nullptr);
      assert(nbWeights == 0 && weights == nullptr);
      mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
          createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
                                 DimsHW(poolingH, poolingW),
                                 Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
                                 Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }),
          nvPluginDeleter);
      return mPluginRPROI.get();
    }
    else
    {
      assert(0);
      return nullptr;
    }
  }

  IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
  {
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "ReshapeCTo2"))
    {
      assert(mPluginRshp2 == nullptr);
      mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
      return mPluginRshp2.get();
    }
    else if (!strcmp(layerName, "ReshapeCTo18"))
    {
      assert(mPluginRshp18 == nullptr);
      mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
      return mPluginRshp18.get();
    }
    else if (!strcmp(layerName, "RPROIFused"))
    {
      assert(mPluginRPROI == nullptr);
      mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
          createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
      return mPluginRPROI.get();
    }
    else
    {
      assert(0);
      return nullptr;
    }
  }

  // caffe parser plugin implementation
  bool isPlugin(const char* name) override
  {
    return (!strcmp(name, "ReshapeCTo2") || !strcmp(name, "ReshapeCTo18") || !strcmp(name, "RPROIFused"));
  }

  // the application has to destroy the plugin when it knows it's safe to do so
  void destroyPlugin()
  {
    mPluginRshp2.release();
    mPluginRshp2 = nullptr;
    mPluginRshp18.release();
    mPluginRshp18 = nullptr;
    mPluginRPROI.release();
    mPluginRPROI = nullptr;
  }

  std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
  std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
  void (*nvPluginDeleter)(INvPlugin*){ [](INvPlugin* ptr) { ptr->destroy(); } };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};

void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo, const int N,
                             const int nmsMaxOut, const int numCls)
{
  float width, height, ctr_x, ctr_y;
  float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
  float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
  for (int i = 0; i < N * nmsMaxOut; ++i)
  {
    width = rois[i * 4 + 2] - rois[i * 4] + 1;
    height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
    ctr_x = rois[i * 4] + 0.5f * width;
    ctr_y = rois[i * 4 + 1] + 0.5f * height;
    deltas_offset = deltas + i * numCls * 4;
    predBBoxes_offset = predBBoxes + i * numCls * 4;
    imInfo_offset = imInfo + i / nmsMaxOut * 3;
    for (int j = 0; j < numCls; ++j)
    {
      dx = deltas_offset[j * 4];
      dy = deltas_offset[j * 4 + 1];
      dw = deltas_offset[j * 4 + 2];
      dh = deltas_offset[j * 4 + 3];
      pred_ctr_x = dx * width + ctr_x;
      pred_ctr_y = dy * height + ctr_y;
      pred_w = exp(dw) * width;
      pred_h = exp(dh) * height;
      predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
      predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
      predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
      predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
    }
  }
}

std::vector<int> nms(std::vector<std::pair<float, int>>& score_index, float* bbox, const int classNum,
                     const int numClasses, const float nms_threshold)
{
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min)
    {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };
  auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
    float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
    float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
    float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };

  std::vector<int> indices;
  for (auto i : score_index)
  {
    const int idx = i.second;
    bool keep = true;
    for (unsigned k = 0; k < indices.size(); ++k)
    {
      if (keep)
      {
        const int kept_idx = indices[k];
        float overlap =
            computeIoU(&bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
        keep = overlap <= nms_threshold;
      }
      else
        break;
    }
    if (keep)
      indices.push_back(idx);
  }
  return indices;
}

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
PluginFactory pluginFactory;
cudaStream_t stream;

int inputIndex0, inputIndex1, outputIndex0, outputIndex1, outputIndex2;
void* buffers[5];

bool is_initialized = false;

void setup(std::string planFilename, float nms_th, float score_th)
{
  nms_threshold = nms_th;
  score_threshold = score_th;

  ifstream planFile(planFilename.c_str());

  if (!planFile.is_open())
  {
    ROS_INFO("Plan Not Found!!!");
    is_initialized = false;
  }
  else
  {
    ROS_INFO("Begin loading plan...");
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    ROS_INFO("*** deserializing");
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), &pluginFactory);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    ROS_INFO("End loading plan...");

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex0 = engine->getBindingIndex(INPUT_BLOB_NAME0);
    inputIndex1 = engine->getBindingIndex(INPUT_BLOB_NAME1);
    outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
    outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
    outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex0], INPUT_C * INPUT_H * INPUT_W * sizeof(float)));    // data
    CHECK(cudaMalloc(&buffers[inputIndex1], IM_INFO_SIZE * sizeof(float)));                   // im_info
    CHECK(cudaMalloc(&buffers[outputIndex0], nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float)));  // bbox_pred
    CHECK(cudaMalloc(&buffers[outputIndex1], nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));   // cls_prob
    CHECK(cudaMalloc(&buffers[outputIndex2], nmsMaxOut * 4 * sizeof(float)));                 // rois

    CHECK(cudaStreamCreate(&stream));

    is_initialized = true;
  }
}

void destroy(void)
{
  if (is_initialized)
  {
    runtime->destroy();
    engine->destroy();
    context->destroy();
    pluginFactory.destroyPlugin();
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex0]));
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
  }
  is_initialized = false;
}

object_msgs::ObjectsInBoxes infer(const sensor_msgs::ImageConstPtr& color_msg)
{
  object_msgs::ObjectsInBoxes bboxes;

  // preprocessing
  cv::Mat image = cv_bridge::toCvShare(color_msg, "bgr8")->image;
  cv::Size imsize = image.size();
  float inputImInfo[3]{ float(imsize.height), float(imsize.width), 1 };
  cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));
  float* inputData = new float[INPUT_C * INPUT_H * INPUT_W];
  for (int i = 0; i < INPUT_H; i++)
  {
    for (int j = 0; j < INPUT_W; j++)
    {
      for (int k = 0; k < INPUT_C; k++)
      {
        const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        inputData[offset] = (float)image.data[offsetCv] - pixelMean[k];
      }
    }
  }

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  auto t_start = chrono::high_resolution_clock::now();
  CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  CHECK(
      cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
  context->enqueue(1, buffers, stream, nullptr);

  // host memory for outputs
  float* outputRois = new float[nmsMaxOut * 4];
  float* outputBboxPred = new float[nmsMaxOut * OUTPUT_BBOX_SIZE];
  float* outputClsProb = new float[nmsMaxOut * OUTPUT_CLS_SIZE];

  CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost,
                        stream));
  cudaStreamSynchronize(stream);

  // predicted bounding boxes
  float* predBBoxes = new float[nmsMaxOut * OUTPUT_BBOX_SIZE];

  bboxTransformInvAndClip(outputRois, outputBboxPred, predBBoxes, inputImInfo, 1, nmsMaxOut, OUTPUT_CLS_SIZE);

  float* bbox = predBBoxes + nmsMaxOut * OUTPUT_BBOX_SIZE;
  float* scores = outputClsProb + nmsMaxOut * OUTPUT_CLS_SIZE;
  for (int c = 1; c < OUTPUT_CLS_SIZE; ++c)  // skip the background
  {
    std::vector<std::pair<float, int>> score_index;
    for (int r = 0; r < nmsMaxOut; ++r)
    {
      if (scores[r * OUTPUT_CLS_SIZE + c] > score_threshold)
      {
        score_index.push_back(std::make_pair(scores[r * OUTPUT_CLS_SIZE + c], r));
        std::stable_sort(score_index.begin(), score_index.end(),
                         [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                           return pair1.first > pair2.first;
                         });
      }
    }

    // apply NMS algorithm
    std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();

    for (unsigned k = 0; k < indices.size(); ++k)
    {
      object_msgs::ObjectInBox BBox;
      BBox.object.object_name = CLASSES[c];
      BBox.object.probability = scores[indices[k] * OUTPUT_CLS_SIZE + c];
      BBox.roi.x_offset = bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4] * imsize.width;
      BBox.roi.y_offset = bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4 + 1] * imsize.height;
      BBox.roi.width = (bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4 + 2] - bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4]) *
                       imsize.width;
      BBox.roi.height =
          (bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4 + 3] - bbox[indices[k] * OUTPUT_BBOX_SIZE + c * 4 + 1]) *
          imsize.height;
      BBox.roi.do_rectify = false;
      bboxes.objects_vector.push_back(BBox);
      bboxes.inference_time_ms = total;
    }
  }

  bboxes.header = color_msg->header;
  return bboxes;
}