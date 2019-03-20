#include "config.hpp"
#include <boost/throw_exception.hpp>
#include <cstdlib>

void config::init() {
  this->add_options(true)
    ("help", "Show help message");
  this->add_options(false)
    ("config,c", boost::program_options::value(&config_files)->composing(),
      "Configuration file. May be passed multiple times.");
}

config::config() : desc_cmd_only("Command-line-only options"), desc_general("General options") {
  init();
}

boost::program_options::options_description_easy_init config::add_options(bool cmd_only) {
  return (cmd_only ? this->desc_cmd_only : this->desc_general).add_options();
}

void config::add_options_description(boost::program_options::options_description& desc, bool cmd_only) {
  (cmd_only ? this->desc_cmd_only : this->desc_general).add(desc);
}

void config::parse(int argc, const char *argv[]) {
  namespace po = boost::program_options;
  boost::program_options::variables_map vm;

  // combine all options
  po::options_description desc_all;
  desc_all.add(this->desc_cmd_only).add(this->desc_general);

  // parse command line
  try {
    po::store(po::parse_command_line(argc, argv, desc_all), vm);
    if (vm.count("help")) {
      std::cout << desc_all << "\n";
      std::exit(0);
    }
  } catch(po::required_option& e) {
    if (vm.count("help")) {
      std::cout << desc_all << "\n";
      std::exit(0);
    }
    throw;
  }

  // parse config files
  if(vm.count("config")) {
    config_files = vm["config"].as<std::vector<std::string>>();
  }
  if(config_files.empty()) { // try default config file
    const char* config_file_name = default_filename.c_str();
    try {
      po::store(
          po::parse_config_file<char>(config_file_name, desc_general), vm);
      po::notify(vm);
    } catch (po::error& e) {
      // ignore exceptions if default file name was not specified by user
    }
  }
  // iterate over config files
  for(size_t i = 0; i < config_files.size(); i++) { // config_files may be extended inside this loop
    const char* config_file_name = config_files[i].c_str();
    po::store(
        po::parse_config_file<char>(config_file_name, desc_general), vm);
    // update config file array if any new files were found
    config_files = vm["config"].as<std::vector<std::string>>();
  }
  po::notify(vm);
  validate();
}

void config::set_default_filename(const std::string filename) {
  default_filename = filename;
}
