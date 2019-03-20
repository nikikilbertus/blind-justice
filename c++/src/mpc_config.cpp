#include "mpc_config.hpp"

mpc_config::mpc_config() {
  namespace po = boost::program_options;
  this->add_options()
    ("party,p", po::value(&party_id)->required(), "Party ID of this process");

  po::options_description desc_server("Server options. These can be passed "
    "multiple times, and each instance specifies an additional server");
  desc_server.add_options()
     ("server.host", po::value(&server_hosts)->composing(), "Server hostname")
     ("server.port", po::value(&server_ports)->composing(), "Server port");
  this->add_options_description(desc_server);
}

void mpc_config::validate() {
  namespace po = boost::program_options;
  if (server_hosts.size() != server_ports.size()) {
    BOOST_THROW_EXCEPTION(po::error("'server.host' and 'server.port' have to be"
        " passed the same number of times"));
  }
  if (party_id < 0) {
      BOOST_THROW_EXCEPTION(po::error("'party' must be non-negative"));
  }
  for (size_t i = 0; i < server_hosts.size(); i++) {
    servers.push_back(server_info(server_hosts[i], server_ports[i]));
  }
  server_hosts.clear();
  server_ports.clear();
  config::validate();
}
