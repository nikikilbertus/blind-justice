#pragma once

#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "server_info.hpp"

class config {
protected:
  std::string default_filename = "config.ini";
  boost::program_options::options_description desc_cmd_only;
  boost::program_options::options_description desc_general;
  std::vector<std::string> config_files;

  void init();

  virtual void validate() {};


public:
  config();
  void parse(int argc, const char *argv[]);

  boost::program_options::options_description_easy_init add_options(bool cmd_only = false);
  void add_options_description(boost::program_options::options_description& desc, bool cmd_only = false);
  void set_default_filename(const std::string filename);
};
