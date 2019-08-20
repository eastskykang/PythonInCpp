//
// Created by donghok on 16.08.19.
//
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <ctime>
#include "python.h"

namespace py = pybind11;
namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // parse arguments
  std::string yaml_path;

  try
  {
    po::options_description desc{"Allowed options"};
    desc.add_options()
        ("help,h", "Help screen")
        ("yaml_path,p", po::value<std::string>(&yaml_path)->required(), "Test configuration yaml path");

    po::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return 0;
    }
  }
  catch (const po::error &ex)
  {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  // start the interpreter and keep it alive
  py::scoped_interpreter guard{};

  auto sys = py::module::import("sys");
  auto append = sys.attr("path").attr("append");
  append(PYTHON_SOURCE_DIR);

  // test class
  auto Test = py::module::import("test_tf").attr("Test");

  // load yaml
  YAML::Node config = YAML::LoadFile(yaml_path);
  const YAML::Node &testSpecs = config["tests"];

  // test spec
  for (auto it = testSpecs.begin(); it != testSpecs.end(); it++) {
    auto testSpec = *it;

    // tag
    auto tag = testSpec["tag"].as<std::string>();

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

    // input matrix
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(batchSize, inputSize);

    std::clock_t start;
    double duration;

    // timer start
    start = std::clock();

    for (int i = 0; i < numStep; i++) {
      auto out = run(input);
    }

    // timer tick
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    // log
    std::cout << "tag         : " << tag << std::endl;
    std::cout << "step        : " << numStep << std::endl;
    std::cout << "elapsed time: " << duration << std::endl;

    // close test
    auto close = test.attr("close");
    close();
  }

//  func();

}