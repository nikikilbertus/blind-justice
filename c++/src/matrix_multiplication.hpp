#pragma once
#include "comm_channel.hpp"
#include "matrix_multiplication.hpp"
#include "mid_layer/DamgardJurikEnc.hpp"
#include "primitives/Prg.hpp"
#include "util/blocking_queue.hpp"
#include "util/prg_adapter.hpp"
#include "util/randomize_matrix.hpp"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/optional.hpp>
extern "C" {
    #include "obliv.h"
    #include "sparse_matrix_multiplication/sparse_matrix_multiplication.h"
}

/**
 * error_info structs for reporting input dimension in exceptions
 */
typedef boost::error_info<struct tag_TRIPLE_L,size_t> error_triple_l;
typedef boost::error_info<struct tag_TRIPLE_M,size_t> error_triple_m;
typedef boost::error_info<struct tag_TRIPLE_N,size_t> error_triple_n;
typedef boost::error_info<struct tag_DIM_A_SIZE1,size_t> error_a_size1;
typedef boost::error_info<struct tag_DIM_A_SIZE2,size_t> error_a_size2;
typedef boost::error_info<struct tag_DIM_B_SIZE1,size_t> error_b_size1;
typedef boost::error_info<struct tag_DIM_B_SIZE2,size_t> error_b_size2;
typedef boost::error_info<struct tag_K_A,size_t> error_k_A;
typedef boost::error_info<struct tag_K_B,size_t> error_k_B;
typedef boost::error_info<struct tag_CHUNK_SIZE,size_t> error_chunk_size;

template<typename T, bool is_shared = false>
class triple_provider {
protected:
  size_t l, m, n;
  int role;

public:
  using triple = std::tuple<boost::numeric::ublas::matrix<T>,
                   boost::numeric::ublas::matrix<T>,
                   boost::numeric::ublas::matrix<T> >;

  triple_provider(size_t l, size_t m, size_t n, int role):
    l(l), m(m), n(n), role(role) {};

  size_t get_l() { return l; }
  size_t get_m() { return m; }
  size_t get_n() { return n; }

  virtual triple get() = 0;
};

template<typename T, bool is_shared = false>
class fake_triple_provider : public virtual triple_provider<T, is_shared> {
using triple = typename triple_provider<T, is_shared>::triple;
private:
  blocking_queue<triple> triples;
  int seed = 34567; // seed random number generator deterministically
  std::mt19937 r;

  triple compute_fake_triple() {
      using namespace boost::numeric::ublas;
      size_t l = this->l, m = this->m, n = this->n;
      int role = this->role;

      std::uniform_int_distribution<T> dist;
      matrix<T> U(l, m), U_mask = is_shared ? matrix<T>(l, m) : zero_matrix<T>(l, m);
      matrix<T> V(m, n), V_mask = is_shared ? matrix<T>(m, n) : zero_matrix<T>(m, n);
      matrix<T> Z_mask(l, n);

      randomize_matrix(r, U);
      randomize_matrix(r, V);
      randomize_matrix(r, Z_mask);
      if(is_shared) {
        randomize_matrix(r, U_mask);
        randomize_matrix(r, V_mask);
      }

      if(role == 0) {
          return std::make_tuple(U - U_mask, V_mask, Z_mask);
      } else {
          return std::make_tuple(U_mask, V - V_mask, prod(U, V) - Z_mask);
      }
  }

public:
  // the `cap` argument allows to use a bounded queue for storing triples,
  fake_triple_provider(size_t l, size_t m, size_t n, int role, ssize_t cap = -1):
    triple_provider<T, is_shared>(l, m, n, role), triples(cap), r(seed) {};
  // blocks if capacity is bounded
  void precompute(size_t num) {
    for(size_t i = 0; i < num; i++) {
      triples.push(this->compute_fake_triple());
    }
  }
  // thread-safe!
  triple get() {
    return triples.pop();
  }
};


template<class MATRIX_A, class MATRIX_B,
  typename T = typename MATRIX_A::value_type, bool is_shared,
  typename std::enable_if<std::is_same<T, typename MATRIX_B::value_type>::value, int>::type = 0>
boost::numeric::ublas::matrix<T> matrix_multiplication(
    const boost::numeric::ublas::matrix_expression<MATRIX_A>& A_in,
    const boost::numeric::ublas::matrix_expression<MATRIX_B>& B_in,
    comm_channel& channel, int role,
    triple_provider<T, is_shared>& triples,
    ssize_t chunk_size_in = -1
) {
  using namespace boost::numeric::ublas;
  const MATRIX_A& A = A_in();
  const MATRIX_B& B = B_in();
  // A : l x m, B: m x n, C: l x n
  size_t l = A.size1(), m = A.size2(), n = B.size2();
  size_t chunk_size = chunk_size_in;
  if(chunk_size_in == -1) {
    chunk_size = l;
  }

  try {
    // check if argument sizes are valid
    if (m != B.size1()) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("Matrix sizes do not match"));
    }
    if(chunk_size != triples.get_l() || triples.get_m() != m || triples.get_n() != n) {
        BOOST_THROW_EXCEPTION(boost::enable_error_info(std::invalid_argument(
            "Triple dimensions do not match matrix dimensions"))
            << error_triple_l(triples.get_l()) << error_triple_m(triples.get_m())
            << error_triple_n(triples.get_n()));
    }
    if(l % chunk_size) { //TODO: allow for different chunk sizes
        BOOST_THROW_EXCEPTION(boost::enable_error_info(std::invalid_argument(
            "`A_in.size1()` must be divisible by `chunk_size`")));
    }

    std::vector<std::function<matrix<T>()>> compute_chunks;
    for(size_t i = 0; i*chunk_size < l; i++) {
        compute_chunks.push_back([&triples, &channel, &B, &A, chunk_size, m, n, role, i]() -> matrix<T> {
        auto chunk_A = subrange(A, i*chunk_size, (i+1)*chunk_size, 0, A.size2());
        // get a multiplication triple
        matrix<T> U, V, Z;
        std::tie(U, V, Z) = triples.get();
        // role 0 sends A - U and receives B - V simultaneously
        // then compute share of the result
        matrix<T, row_major> E = (is_shared || role == 0) * (chunk_A - U), E2(chunk_size, m);
        matrix<T, row_major> F = (is_shared || role == 1) * (B - V), F2(m, n);
        if(role == 0) {
            channel.send_recv(E, F2);
            if(is_shared) {
                channel.send_recv(F, E2);
                E += E2;
            }
            F += F2;
            return prod(E, V) + prod(U, F) + Z;
        } else {
            channel.send_recv(F, E2);
            if(is_shared) {
                channel.send_recv(E, F2);
                F += F2;
            }
            E += E2;
            return prod(E, F) + prod(E, V) + prod(U, F) + Z;
        }
      });
    }
    matrix<T> result = zero_matrix<T>(l, n);
    for(size_t i = 0; i*chunk_size < l; i++) {
      subrange(result, i*chunk_size, (i+1)*chunk_size, 0, result.size2()) = compute_chunks[i]();
    }
    return result;
  } catch(boost::exception& e) {
    e << error_a_size1(A.size1()) << error_a_size2(A.size2())
      << error_b_size1(B.size1()) << error_b_size2(B.size2());
    e << error_chunk_size(chunk_size_in);
    throw;
  }
}


// register BigIntegerCiphertext class for boost::serialization
BOOST_CLASS_EXPORT_IMPLEMENT(BigIntegerCiphertext)

template<class MATRIX_A, class MATRIX_B,
  typename T = typename MATRIX_A::value_type,
  typename std::enable_if<std::is_same<T, typename MATRIX_B::value_type>::value, int>::type = 0>
boost::numeric::ublas::matrix<T> semisparse_matrix_multiplication(
    const boost::numeric::ublas::matrix_expression<MATRIX_A>& A_in,
    const boost::numeric::ublas::matrix_expression<MATRIX_B>& B_in,
    size_t k, comm_channel& channel, int role,
    size_t num_threads = boost::thread::hardware_concurrency(),
    bool fake_offline_phase = false
) {
  using namespace boost::numeric::ublas;
  const MATRIX_A& A = A_in();
  const MATRIX_B& B = B_in();
  // A : l x m, B: m x n, C: l x n
  size_t l = A.size1(), m = A.size2(), n = B.size2();

  try {
    // check if argument sizes are valid
    if (m != B.size1()) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("Matrix sizes do not match"));
    }


    DJKeyGenParameterSpec spec;
    // set up ciphertext packing parameters
    // number of bits needed for padding between elements of type T
    size_t padding_bits = boost::multiprecision::msb(biginteger(k+1)) + 2 // overflows in additions
                          + sizeof(T) * 8 // overflow in multiplication
			                    + 40; // statistical security
    // number of bits needed for an element of type T (with padding for scalar mult)
    size_t packing_shift = sizeof(T) * 8 + padding_bits;
    // number of usable plaintext bits
    size_t max_plaintext_bits = spec.getModulusLength() - 1;
    // number of T elements that fit in a plaintext;
    size_t packing_factor = max_plaintext_bits / packing_shift;
    // number of packs of rows
    size_t num_row_packs = ((l - 1) / packing_factor) + 1;

    // number of threads to use
    if(!num_threads) { num_threads = 1; }
    std::vector<boost::thread> threads(num_threads);
    // duplicate thread-unsafe data structures
    std::vector<shared_ptr<PrgFromOpenSSLAES>> prg_thread(num_threads);
    for(size_t i = 0; i < num_threads; i++) {
        prg_thread[i] = make_shared<PrgFromOpenSSLAES>();
        auto key = prg_thread[i]->generateKey(128);
        prg_thread[i]->setKey(key);
    }
    // compute number of rows to give to each thread
    std::vector<size_t> num_packs_thread(num_threads);
    for(size_t i = 0; i < num_threads; i++) {
        num_packs_thread[i] = num_row_packs / num_threads;
        if(i < num_row_packs % num_threads) { // distribute evenly
            num_packs_thread[i]++;
        }
    }
    // set up encryption parameters and encryption context for each thread
    std::vector<DamgardJurikEnc> enc_thread(num_threads);
    for(size_t i = 0; i < num_threads; i++) {
        enc_thread[i] = DamgardJurikEnc(prg_thread[i]);
        enc_thread[i].setLengthParameter(1);
    }
    if (role == 0) {
        // generate key pair
        auto start = scapi_now();
        auto keypair = enc_thread[0].generateKey(&spec);
        auto public_key = keypair.first;
        auto private_key = keypair.second;
        channel.send(public_key);
        channel.flush();
        // use generated key pair
        for(auto& enc : enc_thread) {
            enc.setKey(public_key, private_key);
        }
        print_elapsed_ms(start, "Key generation");
    } else {
        shared_ptr<PublicKey> public_key;
        channel.recv(public_key);
        for(auto& enc : enc_thread) {
            enc.setKey(public_key, nullptr);
        }
    }

    // storage for the threads' results
    std::vector<matrix<BigIntegerCiphertext>> A_in_enc_thread(num_threads);
    std::vector<matrix<BigIntegerCiphertext>> share_0_enc_thread(num_threads);
    std::vector<matrix<T>> share_1_thread(num_threads);
    for(size_t i = 0; i < num_threads; i++) {
        A_in_enc_thread[i].resize(num_packs_thread[i], m);
        share_0_enc_thread[i].resize(num_packs_thread[i], n);
        share_1_thread[i].resize(num_packs_thread[i] * packing_factor, n);
    }

    // encrypted party 0's input
    matrix<BigIntegerCiphertext> A_in_enc(num_row_packs, m);
    // encrypted party's 0 share of the result
    matrix<BigIntegerCiphertext> share_0_enc(num_row_packs, n);
    // final shares of the result
    matrix<T> share_0(l, n), share_1(l, n);
    if (role == 0) {
      if(!fake_offline_phase) {
        // encrypt matrix
        auto start = scapi_now();
        size_t pack_base = 0;
        for(size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            auto num_row_packs = num_packs_thread[thread_id];
            threads[thread_id] = boost::thread([=,&enc_thread,&A_in_enc_thread,&A](){
                cout << "starting thread " << thread_id << "\n";
                for (size_t col = 0; col < m; col++) {
                    // pack rows together
                    for(size_t row_pack = 0; row_pack < num_row_packs; row_pack++) {
                        biginteger current_pack(0);
                        for (size_t row_in_pack = 0; row_in_pack < packing_factor; row_in_pack++) {
                            size_t row = (pack_base + row_pack) * packing_factor + row_in_pack;
                            if(row >= l) { break; }
                            // insert current row at the correct position
                            current_pack |= biginteger(A(row, col)) << (packing_shift * row_in_pack);
                        }
                        auto plaintext = make_shared<BigIntegerPlainText>(current_pack);
                        A_in_enc_thread[thread_id](row_pack, col) = *dynamic_pointer_cast<BigIntegerCiphertext>(enc_thread[thread_id].encrypt(plaintext));
                    }
                }
            });
            pack_base += num_packs_thread[thread_id];
        }
        pack_base = 0;
        for(size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            threads[thread_id].join();
            for(size_t pack = 0; pack < num_packs_thread[thread_id]; pack++) {
                for(size_t col = 0; col < m; col++) {
                    A_in_enc(pack_base + pack, col) = A_in_enc_thread[thread_id](pack, col);
                }
            }
            pack_base += num_packs_thread[thread_id];
        }
        print_elapsed_ms(start, "Matrix encryption");
        // send encrypted matrix
        start = scapi_now();
        channel.send(A_in_enc);
        channel.flush();
        print_elapsed_ms(start, "Sending encrypted matrix A");
      }
    } else {
        auto start = scapi_now();
        if(!fake_offline_phase) {
          // receive encrypted matrix
          channel.recv(A_in_enc);
          print_elapsed_ms(start, "Receiving encrypted matrix A");
        } else {
          // A_in_enc is already zeroed because of the BigIntegerCiphertext default constructor
        }

        biginteger max_element_padded = (biginteger(1) << (packing_shift - 1)) - 1;
        start = scapi_now();
        size_t pack_base = 0;
        for(size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            auto num_row_packs = num_packs_thread[thread_id];
            threads[thread_id] = boost::thread([=,&prg_thread,&enc_thread,&A_in_enc,&B_in,&share_0_enc_thread,&share_1_thread](){
                for(size_t row_pack = 0; row_pack < num_row_packs; row_pack++) {
                    for (size_t j = 0; j < n; j++) {
                        // construct ciphertext of the share of the inner product
                        // generate random shares and pack them
                        biginteger pack_share_0(0);
                        for(size_t row_in_pack = 0; row_in_pack < packing_factor; row_in_pack++) {
                            size_t row = (pack_base + row_pack) * packing_factor + row_in_pack;
                            if(row >= l) { break; }
                            biginteger current_share = getRandomInRange(0, max_element_padded, prg_thread[thread_id].get());
                            // add current share and its negation to packed shares
                            pack_share_0 |= (max_element_padded - current_share + 1) << (packing_shift * row_in_pack);
                            // save our share
                            current_share &= std::numeric_limits<T>::max(); // use only lowest bits
                            share_1_thread[thread_id](row_pack * packing_factor + row_in_pack, j) = current_share.convert_to<T>();
                        }
                        // encrypt party 0's share pack
                        share_0_enc_thread[thread_id](row_pack, j) = *dynamic_pointer_cast<BigIntegerCiphertext>(enc_thread[thread_id].encrypt(
                            make_shared<BigIntegerPlainText>(pack_share_0)));
                        // compute inner product with our sparse vector
                        for (size_t k = 0; k < m; k++) {
                            if(B(k, j) == 0) { continue; } // exploit sparsity
                            biginteger b(B(k, j));
                            auto c = enc_thread[thread_id].multByConst(&A_in_enc(pack_base + row_pack, k), b);
                            share_0_enc_thread[thread_id](row_pack, j) = *dynamic_pointer_cast<BigIntegerCiphertext>(enc_thread[thread_id].add(&share_0_enc_thread[thread_id](row_pack, j), c.get()));
                        }
                    }
                }
            });
            pack_base += num_packs_thread[thread_id];
        }
        // join threads and merge share matrices
        pack_base = 0;
        for(size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            threads[thread_id].join();
            for(size_t row_pack = 0; row_pack < num_packs_thread[thread_id]; row_pack++) {
                for(size_t j = 0; j < n; j++) {
                    share_0_enc(pack_base + row_pack, j) = share_0_enc_thread[thread_id](row_pack, j);
                    for(size_t row_in_pack = 0; row_in_pack < packing_factor; row_in_pack++) {
                        size_t row = (pack_base + row_pack) * packing_factor + row_in_pack;
                        if(row >= l) { break; }
                        share_1(row, j) = share_1_thread[thread_id](row_pack * packing_factor + row_in_pack, j);
                    }
                }
            }
            pack_base += num_packs_thread[thread_id];
        }
        print_elapsed_ms(start, "Homomorphic multiplication");
    }

    if (role != 0) {
        // send share of result
        auto start = scapi_now();
        channel.send(share_0_enc);
        channel.flush();
        print_elapsed_ms(start, "Sending share of result");
        return share_1;
    } else {
        // receive encrypte share of result and decrypt it
        auto start = scapi_now();
        channel.recv(share_0_enc);
        print_elapsed_ms(start, "Receiving share of result");

        start = scapi_now();
        for (size_t row_pack = 0; row_pack < num_row_packs; row_pack++) {
            for (size_t j = 0; j < n; j++) {
                auto dec = enc_thread[0].decrypt(&share_0_enc(row_pack, j));
                biginteger pack_share_0 = dynamic_pointer_cast<BigIntegerPlainText>(dec)->getX();
                // unpack share for our result
                for(size_t row_in_pack = 0; row_in_pack < packing_factor; row_in_pack++) {
                    size_t row = row_pack * packing_factor + row_in_pack;
                    if(row >= l) { break; }
                    biginteger current_share = pack_share_0 >> (packing_shift * row_in_pack);
                    current_share &= std::numeric_limits<T>::max(); // use only lowest bits
                    share_0(row, j) = current_share.convert_to<T>();
                }
            }
        }
        print_elapsed_ms(start, "Decryption");
        return share_0;
    }
  } catch(boost::exception& e) {
      e << error_a_size1(A.size1()) << error_a_size2(A.size2())
        << error_b_size1(B.size1()) << error_b_size2(B.size2()) << error_k_B(k);
      throw;
  }
}

template<class MATRIX_A, class MATRIX_B,
  typename T = typename MATRIX_A::value_type,
  typename std::enable_if<std::is_same<T, typename MATRIX_B::value_type>::value, int>::type = 0>
boost::numeric::ublas::matrix<T> sparse_matrix_multiplication(
    const boost::numeric::ublas::matrix_expression<MATRIX_A>& A_in,
    const boost::numeric::ublas::matrix_expression<MATRIX_B>& B_in,
    size_t k_A, size_t k_B, // number of non-zero entries in A and B
    comm_channel& channel, int role,
    triple_provider<T, false>& triples,
    ssize_t chunk_size = -1
) {
  using namespace boost::numeric::ublas;
  const MATRIX_A& A = A_in();
  const MATRIX_B& B = B_in();
  // A : l x m, B: m x n, C: l x n
  size_t l = A.size1(), m = A.size2(), n = B.size2();
  try {
    // check if argument sizes are valid
    if (m != B.size1()) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("Matrix sizes do not match"));
    }
    if(k_A > m || k_B > m) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("k_A and k_B can be at most m"));
    }
    if(n > 1) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("Only matrix-vector "
          "multiplication is implemented at this time"));
    }

    // set up protocol transport object for obliv-c
    ProtocolDesc desc;

    // struct channel_transport : ProtocolTransport {
    //     comm_channel *channel;
    // };
    // channel_transport trans;
    // trans.maxParties = 2;
    // // trans.split = nullptr;
    // trans.split = [](ProtocolTransport *trans) -> ProtocolTransport* {
    //     auto self = static_cast<channel_transport*>(trans);
    //     auto trans2 = new channel_transport(*self); // copy struct
    //     trans2->channel = new comm_channel(self->channel->clone());
    //     trans2->cleanup = [](ProtocolTransport *trans2) {
    //         auto self2 = static_cast<channel_transport*>(trans2);
    //         delete self2->channel;
    //         delete self2;
    //     };
    //     return trans2;
    // };
    // trans.send = [](ProtocolTransport* trans, int, const void *data, size_t n) -> int {
    //     auto self = static_cast<channel_transport*>(trans);
    //     self->channel->write(static_cast<const char *>(data), n);
    //     return static_cast<int>(n);
    // };
    // trans.recv = [](ProtocolTransport* trans, int, void *data, size_t n) -> int {
    //     auto self = static_cast<channel_transport*>(trans);
    //     self->channel->read(static_cast<char *>(data), n);
    //     return static_cast<int>(n);
    // };
    // trans.flush = [](ProtocolTransport* trans) -> int {
    //     auto self = static_cast<channel_transport*>(trans);
    //     self->channel->flush();
    //     return 0;
    // };
    // trans.cleanup = [](ProtocolTransport*){}; // do nothing
    // trans.channel = &channel;
    // desc.trans = static_cast<ProtocolTransport*>(&trans);

    auto channel2 = channel.clone();
    int fd = channel2.get_blocking_fd();
    protocolUseTcp2PKeepAlive(&desc, fd, role!=0);
    desc.trans->split = nullptr;

    // if(channel.connect_to_oblivc(desc) == -1) {
    //   BOOST_THROW_EXCEPTION(std::runtime_error("Obliv-C: Connection failed"));
    // }

    setCurrentParty(&desc, role+1);

    // set up inputs for the use in obliv-c
    std::vector<word_t> words_A(k_A); // list of words with non-zero columns
    matrix<T> values_A = zero_matrix<T>(l, k_A); // matrix consisting of non-zero columns
    std::vector<word_index_pair> word_indices_B(k_B); // mapping from words to indices into values_B
    matrix<T> values_B = zero_matrix<T>(k_A + k_B, n); // matrix consisting of non-zero rows of B and additional zeroes (shuffled)
    std::vector<index_t> zeroes_B(k_A); // list of indices corresponding to zeros
    std::vector<index_index_pair> result(k_A);
    auto start = scapi_now();
    if(role == 0) { // set up A's inputs
        size_t count_A = 0;
        for(size_t j = 0; j < m; j++) {
            for(size_t i = 0; i < l; i++) {
                if(A(i, j) != 0 || (m - j) <= (k_A - count_A)) {
                    words_A[count_A] = j;
                    column(values_A, count_A) = column(A, j);
                    count_A++;
                    break;
                }
            }
        }
    } else { // set up B's inputs
        size_t count_B = 0;
        std::vector<T> values_B_temp(k_B);
        for(size_t i = 0; i < m; i++) {
            size_t j = 0;
            if(B(i, j) != 0 || (m - i) <= (k_B - count_B)) {
                word_indices_B[count_B].k = i;
                values_B_temp[count_B] = B(i, j);
                count_B++;
            }
        }
        // get a prg
        auto prg = PrgFromOpenSSLAES();
        auto key = prg.generateKey(128);
        prg.setKey(key);
        prg_adapter<size_t> prg_adapted(prg);
        // generate permutation to shuffle values_B
        std::vector<size_t> permutation(values_B.size1());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), prg_adapted);
        // apply permutation and save indices of values and zeroes
        for(size_t i = 0; i < permutation.size(); i++) {
            size_t index = permutation[i];
            if(i < k_B) {
                values_B(index, 0) = values_B_temp[i];
                word_indices_B[i].v = index;
            } else {
                zeroes_B[i - k_B] = index;
            }
        }
    }
    print_elapsed_ms(start, "Input permuation");
    // set up args struct for sparse_matrix_multiplication
    sparse_matrix_multiplication_args args;
    args.l = l;
    args.m = m;
    args.n = n;
    args.k_A = k_A;
    args.k_B = k_B;
    args.in_A = words_A.data();
    args.in_B = word_indices_B.data();
    args.zeroes_B = zeroes_B.data();
    args.result = result.data();

    // run the protocol
    channel.sync();
    start = scapi_now();
    execYaoProtocol(&desc, sparse_matrix_multiplication, static_cast<void *>(&args));
    cleanupProtocol(&desc);
    channel.sync();
    print_elapsed_ms(start, "Circuit Execution");

    if(role == 0) {
        matrix<T> values_A_permuted = zero_matrix<T>(l, k_A + k_B);
        // permute A according to the result, so that column pair.v_A is at
        // position pair.v_B
        size_t i = 0;
        for(auto pair : result) {
          column(values_A_permuted, pair.v_B) = column(values_A, pair.v_A);
        }
        // now run dense matrix multiplication
        zero_matrix<T> empty_B(k_A + k_B, n);
        return matrix_multiplication(values_A_permuted, empty_B, channel, role, triples, chunk_size);
    } else {
        zero_matrix<T> empty_A(l, k_A + k_B);
        return matrix_multiplication(empty_A, values_B, channel, role, triples, chunk_size);
    }
  } catch(boost::exception& e) {
    e << error_a_size1(A.size1()) << error_a_size2(A.size2())
      << error_b_size1(B.size1()) << error_b_size2(B.size2())
      << error_k_A(k_A) << error_k_B(k_B);
    throw;
  }
}
