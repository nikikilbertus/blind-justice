#pragma once
#include <string>

struct server_info {
  std::string host;
  uint16_t port;

  server_info(std::string &host, uint16_t port) : host(host), port(port) {}

  int compare(const server_info &other) const;
  bool operator==(const server_info &other) const { return (compare(other) == 0); };
  bool operator!=(const server_info &other) const { return (compare(other) != 0); };
  bool operator<=(const server_info &other) const { return (compare(other) <= 0); };
  bool operator>=(const server_info &other) const { return (compare(other) >= 0); };
  bool operator>(const server_info &other) const { return (compare(other) > 0); };
  bool operator<(const server_info &other) const { return (compare(other) < 0); };
};

std::ostream &operator<<(std::ostream &stream, const server_info &server);
