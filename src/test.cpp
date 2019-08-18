//
// Created by donghok on 16.08.19.
//
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include "python.h"

namespace py = pybind11;

int main() {

  // start the interpreter and keep it alive
  py::scoped_interpreter guard{};

  auto sys = py::module::import("sys");
  auto append = sys.attr("path").attr("append");
  append(PYTHON_SOURCE_DIR);

  // test class
  auto Test = py::module::import("test_tf").attr("Test");

  // load yaml
  YAML::Node config = YAML::LoadFile("/home/donghok/git/tfbench/yaml/test.yaml");
  const YAML::Node &testSpecs = config["tests"];

  // test spec
  for (auto it = testSpecs.begin(); it != testSpecs.end(); it++) {
    auto testSpec = *it;

    // device
    auto device = testSpec["device"].as<std::string>();

    // batch size
    int batchSize = testSpec["batch"].as<int>();

    // input size
    int inputSize = testSpec["input"].as<int>();

    // network layer
    auto layerSpecs = py::list();

    for (auto it2 = testSpec["layers"].begin(); it2 != testSpec["layers"].end(); it2++) {
      auto layer = *it2;
      auto type = layer["type"].as<std::string>();

      auto layer_spec = py::dict();

      if (type == "fc") {
        // fully-connected
        layer_spec["type"] = type;
        layer_spec["size"] = layer["size"].as<int>();
        layer_spec["activation"] = layer["activation"].as<std::string>();
      }

      layerSpecs.append(layer_spec);
    }

    // create test
    auto test = Test(layerSpecs, batchSize, inputSize, device);

    // run test
    int numStep = testSpec["step"].as<int>();
    auto run = test.attr("run");

    for (int i = 0; i < numStep; i++) {
      Eigen::MatrixXd inputEigen = Eigen::MatrixXd::Random(batchSize, inputSize);
      run(inputEigen);
    }

    // close test
    auto close = test.attr("close");
    close();
  }

//  func();

}