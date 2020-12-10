# import time
import ray

ray.init()


# @ray.remote
# class Buffer:
#     def __init__(self):
#         self._buffer = []

#     def add(self, n):
#         if len(self._buffer) > 5:
#             self._buffer.pop(0)
#         self._buffer.append(n)

#     def show(self):
#         return self._buffer


# @ray.remote
# def fill_buffer(buffer):
#     while True:
#         time.sleep(random.uniform(0, 1))
#         buffer.add.remote(random.randint(0, 10))


# @ray.remote
# def process_func(buffer):
#     for i in range(3):
#         time.sleep(1)
#         buff = ray.get(buffer.show.remote())
#         print(buff)
#     return "done"


# if __name__ == "__main__":
#     num_workers = 2
#     shared_buffer = Buffer.remote()
#     fillers = [fill_buffer.remote(shared_buffer) for i in range(num_workers)]
#     processor = process_func.remote(shared_buffer)
#     result = ray.get(processor)
#     print(result)
