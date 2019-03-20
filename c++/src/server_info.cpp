#include "server_info.hpp"
#include <ostream>

int server_info::compare(const server_info &other) const {
    int ret = host.compare(other.host);
    if(!ret) { // hostnames equal
        if(port < other.port) return -1;
        if(port > other.port) return 1;
        return 0;
    }
    return ret;
}

std::ostream &operator<<(std::ostream &stream, const server_info &server) {
    return stream << "Server host: " << server.host
                  << ", port: " << server.port;
}
