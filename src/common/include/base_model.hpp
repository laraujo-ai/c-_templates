#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include "onnx_session.hpp"
#include "logger.hpp"
#include <vector>
#include <iostream>

/**
 * @brief Base template class for ONNX inference models.
 *
 * Provides a standardized interface for running inference with ONNX models,
 * handling preprocessing, inference execution, and postprocessing steps.
 *
 * @tparam InputType Type of input data accepted by the model
 * @tparam OutputType Type of output data returned by the model
 */
template<typename InputType, typename OutputType>
class IBaseModel {
public:
    /**
     * @brief Constructs an inference model from an ONNX file.
     * @param model_path Path to the ONNX model file.
     * @param num_threads Number of threads for inference execution.
     */
    IBaseModel(const std::string& model_path, int num_threads, std::string& trt_cache_path);

    virtual ~IBaseModel() = default;

    /**
     * @brief Executes the full inference pipeline on input data.
     * @param input Input data to process.
     * @return Processed output from the model.
     */
    OutputType run(const InputType& input);

protected:
    /**
     * @brief Preprocesses input data into ONNX tensors.
     * @param input Raw input data.
     * @return Vector of ONNX tensors ready for inference.
     */
    virtual std::vector<Ort::Value> preprocess(const InputType& input) = 0;

    /**
     * @brief Runs inference on preprocessed tensors.
     * @param input_tensors Preprocessed input tensors.
     * @return Vector of output tensors from the model.
     */
    virtual std::vector<Ort::Value> infer(std::vector<Ort::Value>& input_tensors);

    /**
     * @brief Postprocesses output tensors into final result.
     * @param output_tensors Raw output tensors from inference.
     * @return Processed output data.
     */
    virtual OutputType postprocess(std::vector<Ort::Value>& output_tensors) = 0;

    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

private:
    /**
     * @brief Extracts input and output names from the ONNX model metadata.
     */
    void extractModelMetadata();
};

template<typename InputType, typename OutputType>
IBaseModel<InputType, OutputType>::IBaseModel(const std::string& model_path, int num_threads, std::string& trt_cache_path)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    ONNXSessionBuilder builder(model_path, num_threads, trt_cache_path);
    session_ = builder.build();
    extractModelMetadata();
}

template<typename InputType, typename OutputType>
OutputType IBaseModel<InputType, OutputType>::run(const InputType& input) {
    std::vector<Ort::Value> input_tensors = preprocess(input);
    std::vector<Ort::Value> output_tensors = infer(input_tensors);
    return postprocess(output_tensors);
}


template<typename InputType, typename OutputType>
std::vector<Ort::Value> IBaseModel<InputType, OutputType>::infer(std::vector<Ort::Value>& input_tensors) {
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;

    for (const auto& name : input_names_) {
        input_names_cstr.push_back(name.c_str());
    }

    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    try {
        return session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime inference error: {}", e.what());
        throw;
    }
}

template<typename InputType, typename OutputType>
void IBaseModel<InputType, OutputType>::extractModelMetadata() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
    }

    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}

#endif // INFERENCER_HPP