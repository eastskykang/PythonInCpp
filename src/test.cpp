//
// Created by donghok on 16.08.19.
//
#include <iostream>
#include <pybind11/embed.h>
#include "python.h"

namespace py = pybind11;

int main() {

  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  auto sys = py::module::import("sys");
  auto append = sys.attr("path").attr("append");
  append(PYTHON_SOURCE_DIR);

  // import our modules
  auto runSession = py::module::import("run_session");
  auto func = runSession.attr("run_session");

  func();

}