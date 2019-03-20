#include "mpc_config.hpp"
#include "party.hpp"
#include "comm_channel.hpp"
#include "logistic_training.hpp"

class logistic_training_config : public virtual mpc_config {
 protected:
  void validate() {
    namespace po = boost::program_options;
    if (num_sensitive_features < 0) {
      BOOST_THROW_EXCEPTION(po::error("'num_sensitive_features' must be positive"));
    }
    if (num_features < 0) {
      BOOST_THROW_EXCEPTION(po::error("'num_features' must be positive"));
    }
    if (num_records < 0) {
      BOOST_THROW_EXCEPTION(po::error("'num_records' must be positive"));
    }
    if (precision < 0) {
      BOOST_THROW_EXCEPTION(po::error("'precision' must be positive"));
    }
    if ((batch_size  & (batch_size  - 1)) != 0 || batch_size == 0) {
      BOOST_THROW_EXCEPTION(po::error("'batch_size' must be a power of 2"));
    }
    if (num_threads < 0) {
      BOOST_THROW_EXCEPTION(po::error("'num_threads' must be positive"));
    }
    mpc_config::validate();
  }

 public:
  ssize_t batch_size;
  ssize_t precision;
  ssize_t num_records;
  ssize_t num_sensitive_features;
  ssize_t num_features;
  ssize_t num_threads;

  logistic_training_config() {
    namespace po = boost::program_options;
    add_options()
      ("batch_size,s", po::value(&batch_size)->required(), "Size of the batches")
      ("precision,f", po::value(&precision)->required(), "Number of bits in fractional part of fixed point encoding")
      ("num_records,n", po::value(&num_records)->required(), "Number of records in training dataset")
      ("num_sensitive_features,p", po::value(&num_sensitive_features)->required(), "Number of sensitive features")
      ("num_features,d", po::value(&num_features)->required(), "Number of features")
      ("num_threads,t", po::value(&num_threads)->default_value(boost::thread::hardware_concurrency()), "Number of threads to use (currently only applies to semi-sparse protocol)")
      ;
    set_default_filename("logistic_training_config.ini");
  }
};

// This assumes memory for A, Z, and Y has been allocated
void gen_synthetic_dataset(matrix<double> &X, matrix<double> &Z, matrix<double> &Y) {
  for (size_t i = 0; i < X.size1(); i++) {
    for (size_t j = 0; j < X.size2(); j++) {
      X(i, j) = 2.5;
    }
    for (size_t j = 0; j < Z.size2(); j++) {
      Z(i, j) = (i < X.size1() / 2) ? 0.0 : 1.0;
    }
    Y(i, 0) = (i < X.size1() / 2) ? -1.0 : 1.0;
  }
}

// This assumes memory for X, Z, Y, X_fp, Z_share, Y_fp has been allocated
template <typename T>
void gen_mpc_input(matrix<double> &X, matrix<double> &Z, matrix<double> &Y,
  matrix<T> &X_fp, matrix<T> &Z_share, matrix<T> &Y_fp, int role, size_t precision) {
  double_matrix2fp(X, X_fp, precision);
  double_matrix2fp(Y, Y_fp, precision);
  if (role == 0) {
    double_matrix2fp(Z, Z_share, precision);
  } else {
    Z_share = zero_matrix<T>(Z_share.size1(), Z_share.size2());
  }
  // In this version party 1 does not have X, Y, only a share of Z,
  // and hence we fix those to 0
  if (role == 1) {
    X_fp = zero_matrix<T>(X_fp.size1(), X_fp.size2());
    Y_fp = zero_matrix<T>(Y_fp.size1(), Y_fp.size2());
  }
}

template <typename T>
int run_party(party &party, logistic_training_config &conf, comm_channel &channel) {
  using namespace boost::numeric::ublas;

  // barrier for timings
  channel.sync(); channel.sync();

  // For party with id 1 X and Y will not be used
  auto start = scapi_now();
  matrix<double> X = zero_matrix<double>(
    conf.num_records, conf.num_features - conf.num_sensitive_features);
  matrix<double> Z = matrix<double>(conf.num_records, conf.num_sensitive_features);
  matrix<double> Y = zero_matrix<double>(conf.num_records, 1);
  matrix<T> X_fp = matrix<T>(X.size1(), X.size2());
  matrix<T> Z_share = matrix<T>(Z.size1(), Z.size2());
  matrix<T> Y_fp = matrix<T>(Y.size1(), Y.size2());

  gen_synthetic_dataset(X, Z, Y);
  gen_mpc_input(X, Z, Y, X_fp, Z_share, Y_fp, party.get_id(), conf.precision);

  cout << "X  = " << X << endl;
  cout << "Z  = " << Z << endl;
  cout << "Y = " << Y << endl;
  cout << "X_fp = " << X_fp << endl;
  cout << "Z_share = " << Z_share << endl;
  cout << "Y_fp = " << Y_fp << endl;

  print_elapsed_ms(start, "Generate data");

  start = scapi_now();
  matrix<T> theta_share = run_logistic_training(
    X_fp, Y_fp, Z_share, channel, party.get_id(), conf.batch_size, conf.num_threads, conf.precision);
  print_elapsed_ms(start, "Logistic regression training");
  return 0;
}

int main(int argc, const char *argv[]) {
  using T = uint32_t;
  logistic_training_config conf;
  try {
      conf.parse(argc, argv);
  } catch (boost::program_options::error &e) {
      std::cerr << e.what() << "\n";
      return 1;
  }
  if(conf.party_id != 0 && conf.party_id != 1) {
        std::cerr << "party must be 0 or 1\n";
        return 1;
  }
  if(conf.num_features <= conf.num_sensitive_features) {
        std::cerr << "Number of features must be bigger than number of sensitive features\n";
        return 1;
  }
  party party(conf);
  cout << party.get_id() << endl;
  auto channel = party.connect_to(1 - party.get_id());

  run_party<T>(party, conf, channel);
}


/*
Command lines:

bin/logistic_training_timing --batch_size 2 --num_features 10 --num_records 100 --num_sensitive_features 6 --party 1 --num_threads 4  -c src/cmd/servers_conf.ini --precision 12 &
bin/logistic_training_timing --batch_size 2 --num_features 10 --num_records 100 --num_sensitive_features 6 --party 0 --num_threads 4  -c src/cmd/servers_conf.ini --precision 12

bin/logistic_training_timing --batch_size 1024 --num_features 10 --num_records 100000 --num_sensitive_features 6 --party 1 --num_threads 4  -c src/cmd/servers_conf.ini --precision 12 &
bin/logistic_training_timing --batch_size 1024 --num_features 10 --num_records 100000 --num_sensitive_features 6 --party 0 --num_threads 4  -c src/cmd/servers_conf.ini --precision 12

*/