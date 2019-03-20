#include "comm_channel.hpp"
#include "matrix_multiplication.hpp"
#include "fp.hpp"
#include <boost/range/irange.hpp>
#include <string>
extern "C" {
    #include "obliv.h"
    #include "sigmoid_approx/secureml_sigmoid.h"
    #include "select_theta/select_theta.h"
}

using namespace boost::numeric::ublas;

/*This is used for debugging purposes,
to reconstruct and print an intermediate
secretly shared value
*/
template <typename T>
void print_shared_matrix(
    matrix<T> A, string name, comm_channel &channel, int role, size_t precision) {
    matrix<T> A2;
    // exchange shares for checking result
    channel.send_recv(A, A2);
    cout << name << "(fp)" << A + A2 << endl;
    matrix<double> M = matrix<double>(A.size1(), A.size2());
    matrix<T> M_fp = matrix<T>(A.size1(), A.size2());
    M_fp = A + A2;
    fp_matrix2double(M_fp, M, precision);
    cout << name << "(double)" << M << endl;
}

/*This is used for debugging purposes,
to reconstruct and print an intermediate
secretly shared value
*/
template <typename T>
void print_shared_multiplication(
    matrix<T> A, matrix<T> B, matrix<T> C, comm_channel& channel, int role, size_t precision) {
    matrix<T> A2, B2, C2;
    print_shared_matrix(A, "A", channel, role, precision);
    print_shared_matrix(B, "B", channel, role, precision);
    print_shared_matrix(C, "C", channel, role, precision);
}

/*
Multiply a shared matrix by a public escalar
*/
template <typename T>
matrix<T> fixpoint_matrix_multiplication_by_scalar(matrix<T> M_share, T lambda) {
    matrix<T> res(M_share.size1(), M_share.size2());
    for (size_t i = 0; i < M_share.size1(); ++i) {
        for (size_t j = 0; j < M_share.size2(); ++j) {
            res(i, j) = M_share(i, j) * lambda;
        }
    }
    return res;
}

/*
This corresponds to having the parties rescale their
shares locally, and may result in an erroneous
result with some probability (see secureML paper).
*/
template <typename T>
void lossy_local_rescale(matrix<T> M_share, size_t scale) {
    for (size_t i = 0; i < M_share.size1(); ++i) {
        for (size_t j = 0; j < M_share.size2(); ++j) {
            // TODO make sure that this preserves the sign bit
            // TODO: define sT and the signed verstion of T, so that
            // we don't have int32_t in the code
            M_share(i, j) = (T)(((int32_t)M_share(i, j)) >> scale);
        }
    }
}

/*
Secure multiplication of two matrices A, B, where
A is held by party 1 and B is shared additively between parties 1 and 2.
The matrices encode fixpoint numbers which need to be rescaled after the multiplication.
*/
template <typename T>
matrix<T> fixpoint_matrix_multiplication(
    matrix<T> A, matrix<T> B, comm_channel& channel, int role,
        size_t precision, fake_triple_provider<T, false> &triples) {
    matrix<T> C = matrix<T>(A.size1(), B.size2());
    size_t chunk_size = -1;
    if (role == 0) {
        C = matrix_multiplication(
            A, zero_matrix<T>(B.size1(), B.size2()), channel, role, triples, chunk_size);
        // TODO: Consider rescaling once per iteration
        lossy_local_rescale(C, precision);
        // This is because A is held only by party 0, and hence we have to do
        // A(B_1 + B_2) = AB_1 (locally) + AB_2 (securely)
        C = C + prod(A, B);
    } else {
        C = matrix_multiplication(
            zero_matrix<T>(A.size1(), A.size2()), B, channel, role, triples, chunk_size);
        // TODO: Consider rescaling once per iteration
        lossy_local_rescale(C, precision);
    }
    print_shared_multiplication(A, B, C, channel, role, precision);
    return C;
}


template <typename T>
matrix<T> compute_activation(matrix<T> M, comm_channel& channel, int role, size_t precision) {
    // M is a matrix of dimension (batch_size, 1)
    // check if argument sizes are valid
    if (M.size2() != 1) {
        BOOST_THROW_EXCEPTION(std::invalid_argument("M is expected to be an (n, 1) matrix"));
    }

    // set up protocol transport object for obliv-c
    ProtocolDesc desc;

    auto channel2 = channel.clone();
    int fd = channel2.get_blocking_fd();
    protocolUseTcp2PKeepAlive(&desc, fd, role!=0);
    desc.trans->split = nullptr;

    // if(channel.connect_to_oblivc(desc) == -1) {
    //   BOOST_THROW_EXCEPTION(std::runtime_error("Obliv-C: Connection failed"));
    // }

    // oblivc indexes parties with 1, 2
    setCurrentParty(&desc, role + 1);

    // set up inputs for the use in obliv-c
    std::vector<T> v_share_0_plus_half(M.size1());
    std::vector<T> v_share_0_minus_half(M.size1());
    std::vector<T> v_share_1(M.size1());
    std::vector<uint8_t> result_share_0(M.size1());
    std::vector<uint8_t> result_share_1(M.size1());
    auto start = scapi_now();
    uint32_t one_half;
    double2fp(0.5, one_half, precision);
    for(size_t i = 0; i < M.size1(); i++) {
        if (role == 0) {
            v_share_0_plus_half[i] = M(i, 0) + one_half;
            v_share_0_minus_half[i] = M(i, 0) - one_half;
        } else {
            v_share_1[i] = M(i, 0);
            result_share_1[i] = 0;
        }
    }
    print_elapsed_ms(start, "Setting up inputs to garbled circuit");
    secureml_sigmoid_args args;
    args.n = M.size1();
    args.v_share_0_plus_half = v_share_0_plus_half.data();
    args.v_share_0_minus_half = v_share_0_minus_half.data();
    args.v_share_1 = v_share_1.data();
    args.result_share_0 = result_share_0.data();
    args.result_share_1 = result_share_1.data();

    // run the protocol
    channel.sync();
    start = scapi_now();
    execYaoProtocol(&desc, secureml_sigmoid, static_cast<void *>(&args));
    cleanupProtocol(&desc);
    channel.sync();
    print_elapsed_ms(start, "Garbled Circuit Execution");
    matrix<T> res(M.size1(), 1);
    for (size_t i = 0; i < res.size1(); ++i) {
        res(i, 0) = (role == 0) ? (uint32_t)args.result_share_0[i] : (uint32_t)args.result_share_1[i];
    }
    if (role == 0) {
        //cout << "share0  = " << res << endl;
    } else {
        //cout << "share1  = " << res << endl;
    }
    return res;
}

template <typename T>
matrix<T> run_logistic_training(
    matrix<T> X, matrix<T> Y, matrix<T> Z_share, comm_channel& channel, int role,
    ssize_t batch_size, ssize_t num_threads, size_t precision) {

    matrix<T> theta_share = zero_matrix<T>(X.size2(), 1);
    // For 1 epoch num_iters is X.size1() / batch_size
    int num_iters = X.size1() / batch_size;
    // At each iteration we'll compute
    // B^T * (f(B*theta) - Y), where
    // B is the batch for the current iter and
    // f is an approximation of sigmoid

    // First precompute all 'fake' triples for matrix vector multiplications
    fake_triple_provider<T, false> triples_fw(batch_size, X.size2(), 1, role);
    fake_triple_provider<T, false> triples_bw(X.size2(), batch_size, 1, role);
    auto start = scapi_now();
    triples_fw.precompute(num_iters);
    triples_bw.precompute(num_iters);
    print_elapsed_ms(start, "Fake Triple Generation");
    for (int i : boost::irange(0, num_iters)) {
        cout << "Iteration " << i << endl;
        // B is the batch for the current iter
        matrix<T> B = project(
            X, range(batch_size*i, batch_size*i + batch_size), range(0, X.size2()));
        matrix<T> Y_B = project(
            Y, range(batch_size*i, batch_size*i + batch_size), range(0, Y.size2()));

        // We first compute a share of B*theta using
        // dense matrix multiplication wirh precomputed triples

        matrix<T> B_theta_share;
        start = scapi_now();
        B_theta_share = fixpoint_matrix_multiplication(
            B, theta_share, channel, role, precision, triples_fw);
        // cout << B_theta_share << endl;
        print_elapsed_ms(start, "Computation of B * theta");
        // We now evaluate f(B*theta) and have the parties obtain shares
        // of that value using a garbled circuit
        matrix<T> f_B_theta_share = compute_activation(B_theta_share, channel, role, precision);
        // Party 0 substracts Y_B from her share, as she holds Y_B in the clear
        f_B_theta_share = (role == 0) ? f_B_theta_share - Y_B : f_B_theta_share;
        print_shared_matrix(f_B_theta_share, "f(B*theta))", channel, role, precision);

        // We have to do one final matrix
        // vector multiplication (B^T * f_B_theta) and a division
        // by the batch size (which is a power of two) to compute the gradient
        matrix<T> gradient_share;
        start = scapi_now();
        gradient_share = fixpoint_matrix_multiplication(
            (matrix<T>)trans(B), f_B_theta_share, channel, role, precision, triples_bw);
        lossy_local_rescale(gradient_share, log2(batch_size));
        print_elapsed_ms(start, "Computation of B^T * (f(B * theta) - Y)");

        double eta = 0.1;
        T eta_fp;
        double2fp(eta, eta_fp, precision);
        theta_share = theta_share + fixpoint_matrix_multiplication_by_scalar(
                    gradient_share, eta_fp);
        print_shared_matrix(theta_share, "theta", channel, role, precision);

    }

    return theta_share;
}
