#pragma once
#include "party.hpp"
#include "comm/Comm.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <iostream>

/**
 * calls `call`, catching exceptiony of type `exception_type`, wrapping them
 * in a boost::exception and adding the error message of this channel's stream
 */
#define COMM_CHANNEL_WRAP_EXCEPTION(call, exception_type) \
    do { \
        try { \
            call; \
        } catch(exception_type &ex) { \
            BOOST_THROW_EXCEPTION(boost::enable_error_info(ex) \
                << stream_error(stream->error().message())); \
        } \
    } while(0)


/**
 * A bidirectional connection that can be used both as a CommParty in libscapi
 * calls, and using the boost serialization library.
 */
class comm_channel: public CommParty {
public:
    comm_channel(std::unique_ptr<boost::asio::ip::tcp::iostream>&& s,
        party& p, int peer_id);

    // no copy constructor: since copying a channel means establishing a
    // new connection, this should be done explicitly using clone()
    comm_channel(comm_channel &other) = delete;

    // move constructor
    comm_channel(comm_channel &&other): p(other.p), id(other.id),
        peer_id(other.peer_id), stream(std::move(other.stream)),
        iarchive(std::move(other.iarchive)),
        oarchive(std::move(other.oarchive)), need_flush(other.need_flush) {};

    // connection happens on stream construction -> do nothing on join
    void join(int, int) {};

    // write & read functions for direct binary access
    void write(const char *data, size_t size);
    void read(char* buffer, size_t size);
    // write & read functions for CommParty interface
    void write(const byte *data, int size);
    size_t read(byte* buffer, int size);
    // send and receive objects via boost serialization
    template<typename T> void send(T &obj) {
        need_flush = true;
        COMM_CHANNEL_WRAP_EXCEPTION(
            *oarchive & obj,
            boost::archive::archive_exception
        );
    }
    template<typename T> void recv(T &obj) {
        flush_if_needed();
        COMM_CHANNEL_WRAP_EXCEPTION(
            *iarchive & obj,
            boost::archive::archive_exception
        );
    }
    // send from `to_send` and receive into `to_recv` simultaneously
    template<typename S, typename T> void send_recv(S& to_send, T& to_recv) {
        if(!twin) {
          twin = std::unique_ptr<comm_channel>(new comm_channel(clone()));
        }
        comm_channel& channel2 = *twin;
        if(get_id() < get_peer_id()) { // lower ID sends on channel2
            boost::thread t([&to_send, &channel2]() {
                channel2.send(to_send);
                channel2.flush();
            });
            boost::thread_guard<> g(t); // join t when leaving scope
            recv(to_recv);
        } else { // higher ID receives on channel2
            boost::thread t([&to_recv, &channel2]() {
                channel2.recv(to_recv);
            });
            boost::thread_guard<> g(t); // join t when leaving scope
            send(to_send);
            flush();
        }
    }

    // flush underlying stream
    void flush();

    // ensure everything up to this point was sent/received
    void sync();

    // create a second comm_channel from this one; establishes a new connection
    comm_channel clone();

    int connect_to_oblivc(ProtocolDesc& pd) {
      return p.connect_to_oblivc(pd, peer_id);
    }

    // returns a blocking file descriptor of the underlying socket
    int get_blocking_fd() {
      stream->rdbuf()->native_non_blocking(false);
      return stream->rdbuf()->native_handle();
    }

    // getters for both parties' IDs
    int get_id() { return id; }
    int get_peer_id() { return peer_id; }

    /**
     * Error info type for exceptions thrown from class methods
     */
    typedef party::error_peer_id error_peer_id;
    typedef party::error_my_id error_my_id;
    typedef party::error_num_servers error_num_servers;
    typedef boost::error_info<struct tag_STREAM_ERROR,std::string> stream_error;

private:
    party& p;
    int id;
    int peer_id;
    std::unique_ptr<boost::asio::ip::tcp::iostream> stream;
    std::unique_ptr<boost::archive::binary_iarchive> iarchive;
    std::unique_ptr<boost::archive::binary_oarchive> oarchive;
    std::unique_ptr<comm_channel> twin; // used for sending and receiving simultaneously;
    // TODO: somehow allow for simultaneous reading and writing on the _same_ socket
    bool need_flush;

    void flush_if_needed() {
        if(need_flush) { flush(); }
    }
};
