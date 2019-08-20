//
// Created by donghok on 16.08.19.
//
#include <iostream>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <ctime>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "python.h"

namespace py = pybind11;
namespace po = boost::program_options;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


int main(int argc, const char *argv[]) {

  // parse arguments
  std::string yamlPath;
  bool loopTime = false;
  bool csv = false;

  try
  {
    po::options_description desc{"Allowed options"};
    desc.add_options()
        ("help,h", "Help screen")
        ("yaml_path,p", po::value<std::string>(&yamlPath)->required(), "Test configuration yaml path")
        ("loop_time,l", "Log loop timer")
        ("csv,c", "Create .csv file");

    po::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return 0;
    }

    loopTime = vm.count("loop_time");
    csv = vm.count("csv");
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
  YAML::Node config = YAML::LoadFile(yamlPath);
  const YAML::Node &testSpecs = config["tests"];

  // csv
  std::ofstream csvfile;
  if (csv) {

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%d-%m-%H-%M-%S.csv");

    csvfile.open(oss.str());
    csvfile << "tag, step, elapsed time,\n";
  }

  // test spec
  for (auto it = testSpecs.begin(); it != testSpecs.end(); it++) {
    auto testSpec = *it;

    // tag
    const auto tag = testSpec["tag"].as<std::string>();

    // device
    const auto device = testSpec["device"].as<std::string>();

    // thread
    const int intraThread = testSpec["intraThread"].as<int>();
    const int interThread = testSpec["interThread"].as<int>();

    // batch size
    const int batchSize = testSpec["batch"].as<int>();

    // input size
    const int inputSize = testSpec["input"].as<int>();

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
    auto test = Test(layerSpecs, batchSize, inputSize, device, intraThread, interThread);

    // run test
    int numStep = testSpec["step"].as<int>();
    auto run = test.attr("run");

    // input matrix
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(batchSize, inputSize);
//    RowMatrixXd input = Eigen::MatrixXd::Random(batchSize, inputSize);

    std::clock_t start;
    double duration;

    std::clock_t loop_clock;
    std::vector<double> loop_durations;
    loop_durations.reserve(numStep);

    // timer tic
    start = std::clock();

    for (int i = 0; i < numStep; i++) {
      loop_clock = std::clock();
      auto out = run(input);
      loop_durations.push_back(( std::clock() - loop_clock ) / (double) CLOCKS_PER_SEC);
    }

    // timer toc
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    // log
    std::cout << "tag         : " << tag << std::endl;
    std::cout << "step        : " << numStep << std::endl;
    std::cout << "intra thread: " << intraThread << std::endl;
    std::cout << "inter thread: " << interThread << std::endl;
    std::cout << "elapsed time: " << duration << std::endl;

    if (loopTime) {
      std::cout << "loop time   : " << std::endl;
      for (int i = 0; i < loop_durations.size(); i++)
        std::cout << loop_durations[i] << std::endl;
    }

    if (csv) {
      csvfile << tag << "," << numStep << "," << duration << ",\n";
    }

    // close test
    auto close = test.attr("close");
    close();
  }

  if (csv) {
    csvfile.close();
  }

}