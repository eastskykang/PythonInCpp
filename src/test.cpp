//
// Created by donghok on 16.08.19.
//
#include <iostream>
#include <pybind11/embed.h>  // python interpreter

namespace py = pybind11;

int main() {

  py::scoped_interpreter guard{}; // start the interpreter and keep it alive


  auto sys = py::module::import("sys");
  py::print(sys.attr("executable"));

//  auto runSession = py::module::import("run_session");
//
//  auto func = module.attr("run_session");
//
//  func();


}