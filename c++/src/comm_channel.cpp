#include "comm_channel.hpp"

// constructor called from party::connect_to
comm_channel::comm_channel(
    std::unique_ptr<boost::asio::ip::tcp::iostream>&& s,
    party& p, int peer_id
) :
    p(p), id(p.get_id()), stream(std::move(s)), need_flush(false)
{
    oarchive = std::unique_ptr<boost::archive::binary_oarchive>(
        new boost::archive::binary_oarchive(*stream, boost::archive::no_codecvt));
    flush();
    iarchive = std::unique_ptr<boost::archive::binary_iarchive>(
        new boost::archive::binary_iarchive(*stream, boost::archive::no_codecvt));
    if(peer_id == -1) {
        recv(peer_id); // read id of remote
    } else {
        send(this->id); // send own ID to remote
        flush();
    }
    if(peer_id < 0 || peer_id == id) {
        BOOST_THROW_EXCEPTION(boost::enable_error_info(std::invalid_argument(
            "invalid peer_id")) << error_num_servers(p.get_num_servers())
            << error_my_id(id) << error_peer_id(peer_id));
    }
    this->peer_id = peer_id;
};

// write & read functions for direct binary access
void comm_channel::write(const char *data, size_t size) {
    COMM_CHANNEL_WRAP_EXCEPTION(
        oarchive->save_binary(data, size),
        boost::archive::archive_exception
    );
    need_flush = true;
}
void comm_channel::read(char* buffer, size_t size) {
    flush_if_needed();
    COMM_CHANNEL_WRAP_EXCEPTION(
        iarchive->load_binary(buffer, size),
        boost::archive::archive_exception
    );
}
// write & read functions for CommParty interface
void comm_channel::write(const byte *data, int size) {
    write(reinterpret_cast<const char *>(data), size);
}
size_t comm_channel::read(byte* buffer, int size) {
    read(reinterpret_cast<char *>(buffer), size);
    return size;
}

// create a second comm_channel from this one; establishes a new connection
comm_channel comm_channel::clone() {
    tcp::no_delay no_delay;
    stream->rdbuf()->get_option(no_delay);
    flush_if_needed();
    return p.connect_to(peer_id, no_delay.value());
}

// flush underlying stream
void comm_channel::flush() {
    COMM_CHANNEL_WRAP_EXCEPTION(
        stream->flush(),
        std::ios_base::failure
    );
    need_flush = false;
}

void comm_channel::sync() {
    int a = 0, b;
    send_recv(a, b);
    // // sequentially send and receive to save the reconnection that would
    // // occur with send_recv
    // if(this->get_id() < this->get_peer_id()) {
    //   send(a);
    //   flush();
    //   recv(b);
    // } else {
    //   recv(b);
    //   send(a);
    //   flush();
    // }
}
