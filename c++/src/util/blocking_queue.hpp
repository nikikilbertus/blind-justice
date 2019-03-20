// StackOverflow answer by Dietmar Kühl, licensed under cc by-sa 3.0
// https://stackoverflow.com/a/12805690
// adapted to support capacity
#include <mutex>
#include <condition_variable>
#include <deque>

template <typename T>
class blocking_queue
{
private:
    std::mutex              d_mutex;
    std::condition_variable d_condition_not_empty;
    std::condition_variable d_condition_not_full;
    std::deque<T>           d_queue;
    ssize_t                 d_capacity;

    bool empty() const { return d_queue.empty(); }
    bool full() const { return  d_capacity != -1 && d_queue.size() == d_capacity; }
public:
  blocking_queue(ssize_t cap = -1): d_capacity(cap) {}
  void push(T const& value) {
      {
        std::unique_lock<std::mutex> lock(this->d_mutex);
        this->d_condition_not_full.wait(lock, [=]{ return !full(); });
        d_queue.push_front(value);
      }
      this->d_condition_not_empty.notify_one();
  }
  T pop() {
      T rc;
      {
        std::unique_lock<std::mutex> lock(this->d_mutex);
        this->d_condition_not_empty.wait(lock, [=]{ return !empty(); });
        rc = std::move(this->d_queue.back());
        this->d_queue.pop_back();
      }
      this->d_condition_not_full.notify_one();
      return rc;
  }
};
