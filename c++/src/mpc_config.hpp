#pragma once
#include "config.hpp"

class mpc_config : public virtual config {
private:
  std::vector<std::string> server_hosts;
  std::vector<uint16_t> server_ports;

protected:
  virtual void validate();

public:
  std::vector<server_info> servers;
  int party_id;

  mpc_config();
};
