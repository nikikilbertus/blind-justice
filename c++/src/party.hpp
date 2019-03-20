#pragma once
#include "mpc_config.hpp"
#include <boost/asio.hpp>
#include <boost/exception/all.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_guard.hpp>
#include <mutex>
#include <string>
extern "C" {
  #include "obliv.h"
}

// forward_declaration to resolve circular reference between party and comm_channel
// comm_channel.h gets included after declaration of class party below
class comm_channel;

// represents a participant in an MPC protocol
class party {
public:
    party(mpc_config& conf) : servers(conf.servers), id(conf.party_id),
        io_thread(boost::bind(&boost::asio::io_service::run, &io_service)),
        resolver(io_service), io_guard(io_thread), pending(servers.size() + 1)
        {};

    /**
     * Establishes a connection to a party with ID party_id and returns a pointer to an
     * object representing that connection.
     * Suitable for passing to SCAPI calls
     */
    comm_channel connect_to(int peer_id, bool tcp_nodelay = true, int sleep_time = 500, int num_tries = -1);

    int connect_to_oblivc(ProtocolDesc &pd, int peer_id, int sleep_time = 500, int num_tries = -1);

    /**
     * Return this party's ID
     */
    int get_id() { return id; }

    /**
     * Return the number of servers
     */
    int get_num_servers() { return servers.size(); }

    /**
     * Error info types for exceptions thrown from class methods
     */
    typedef boost::error_info<struct tag_PEER_ID, int> error_peer_id;
    typedef boost::error_info<struct tag_MY_ID, int> error_my_id;
    typedef boost::error_info<struct tag_NUM_SERVERS, size_t> error_num_servers;

    static const int ANY_PEER = -1;

private:
    std::vector<server_info> servers; // list of all servers sorted by party id
    int id; // this party's ID
    boost::asio::io_service io_service; // IO context for this party
    boost::thread io_thread; // runs io.run()
    boost::asio::ip::tcp::resolver resolver;
    boost::thread_guard<> io_guard; // joins io_thread when destroyed
    std::mutex connection_mutex; // allow only one concurrent call to connect_to
    std::vector<std::vector<comm_channel>> pending;
};

#include "comm_channel.hpp"
