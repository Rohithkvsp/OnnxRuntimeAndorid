//
//  onnxruntime_inference.cpp
//  onnxRuntimeInference
//
//  Created by rohith kvsp on 9/13/20.
//  Copyright © 2020 rohith kvsp. All rights reserved.
//

#include "onnxruntime_inference.h"
#include "logs.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"


Inference::Inference(std::unique_ptr<Ort::Env>& env, const char* modelpath, const char* labelfilepath, int img_height, int img_width) : env_(std::move(env)), modelpath_(modelpath), labelfilepath_(labelfilepath),
                                                                                                                       img_height_(img_height), img_width_(img_width) {
    LOGD("model path  %s ", modelpath_);
    LOGD("label file path %s ", labelfilepath_);
    LOGD("Input image height  %d ", img_height_);
    LOGD("Input image width %d ", img_width_);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_  = std::make_unique<Ort::Session>(*env_.get(), modelpath_, session_options);
    printNodes();
    createInputBuffer();
    readLabelsFile( std::string(labelfilepath_), labels);


}

void Inference::createInputBuffer()
{
    LOGD("creating input data buffer of size =  %lu", input_tensor_size);
    input_data_chw  = std::make_unique<float[]>(input_tensor_size);
    normalized = std::make_unique<float[]>(input_tensor_size);
    
    ouput_tensor_size = output_node_dims[output_node_dims.size()-1];
    LOGD("creating ouput data buffer of size =  %lu", ouput_tensor_size);
    softmax_output = std::make_unique<float[]>(ouput_tensor_size);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_chw.get(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());


}

void Inference::printNodes() {
     Ort::AllocatorWithDefaultOptions allocator;
        
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names.reserve(num_input_nodes);
    
    
    LOGD("Number of input =  %zu",num_input_nodes);
    
    for (int i = 0; i < num_input_nodes; i++){
        char* input_name = session_->GetInputName(i, allocator);
        LOGD("Input %d : name = %s", i, input_name);
        input_node_names[i] = input_name;
        
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        LOGD("Input %d : type = %d", i, type);
        
        input_node_dims = tensor_info.GetShape();
        LOGD("Input %d : num_dims=%zu", i, input_node_dims.size());
        
        input_tensor_size = 1;
        for (int j = 0;j < input_node_dims.size(); j++)
        {
            LOGD("Input %d : dim %d = %ld",i,j,input_node_dims[j]);
            input_tensor_size *= input_node_dims[j];
        }
    }
    
    
    
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names.reserve(num_output_nodes);
    
    
    for (int i = 0; i < num_output_nodes; i++){
        char* output_name = session_->GetOutputName(i, allocator);
        LOGD("Output %d : name = %s", i, output_name);
        output_node_names[i] = output_name;
        
        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        LOGD("Output %d : type = %d", i, type);
        
        output_node_dims = tensor_info.GetShape();
        LOGD("Output %d : num_dims=%zu", i, output_node_dims.size());
        
        for (int j = 0;j < output_node_dims.size(); j++)
        {
             LOGD("Output %d : dim %d = %ld ",i,j,output_node_dims[j]);
        }
    }
}


void Inference::run(uint8_t* pixels){

    auto start = std::chrono::high_resolution_clock::now();
    
    preprocess(pixels, img_height_, img_width_, 4, normalized.get(), {0.485f, 0.456f, 0.406f}, {0.229, 0.224, 0.225});
    HWCtoCHW(normalized.get(), img_height_, img_width_, 3, input_data_chw.get());
    
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    
    
   softmax(floatarr, softmax_output.get(),ouput_tensor_size );
       
   std::vector<std::pair<float, int>> results;
   
   KTopResults(softmax_output.get(), ouput_tensor_size  ,5, results);
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

   predicted_labels = "";
   for(int i=0;i<3;i++)
   {
       LOGD("%f , %d %s" , results[i].first, results[i].second, labels[results[i].second].c_str());
       predicted_labels += labels[results[i].second] +" : "+std::to_string(results[i].first)+"\n";

   }
    predicted_labels += " Run time "+std::to_string(duration.count())+" milli seconds";


    LOGD("Time to run: %" PRId64 " milli seconds \n",duration.count());

}


std::string Inference::getPredictedlabels()
{
    return predicted_labels;
}

Inference::~Inference(){
    delete modelpath_;
    delete labelfilepath_;
}
