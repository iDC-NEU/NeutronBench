# Importing the essential libraries
import psutil
import GPUtil
# Testing the psutil library for both CPU and RAM performance details
import time



def get_cpu_info(time_start=0):
  cpu_usage = psutil.cpu_percent()
  mem_usage = psutil.virtual_memory().percent
  print(f'CPU_info: curr_time {round(time.time() - time_start, 4)}s cpu_usage {cpu_usage}, cpu_mem_usage {mem_usage}')


def get_all_gpu_info(occupy_mem, time_start=0):
  gpu_list = GPUtil.getGPUs()
  for i, gpu in enumerate(gpu_list):
    # print("GPU ID：", gpu.id)
    # print("GPU名称：", gpu.name)
    # print("总内存：", gpu.memoryTotal)
    # print("已使用内存：", gpu.memoryUsed)
    # print("计算占用：", gpu.load * 100, "%")
    gid = gpu.id
    mem_use = gpu.memoryUsed
    mem_total = gpu.memoryTotal
    mem_usage = round(mem_use * 100 / mem_total, 2)
    print(f'GPU_info: curr_time {round(time.time() - time_start, 4)}s  gpumem{gid}: ({mem_use}/{mem_total}) gpu_mem_usage{gid}: {mem_usage}%  gpu_compute_usage{gid}: {gpu.load * 100}%')


def get_gpu_info(gid, time_start=0,):
  gpu_list = GPUtil.getGPUs()
  mem_use = float(gpu_list[gid].memoryUsed)
  mem_total = float(gpu_list[gid].memoryTotal)
  mem_usage = round(mem_use * 100 / mem_total, 2)
  load = round(gpu_list[gid].load * 100, 2)
  print(f'GPU_info: curr_time {round(time.time() - time_start, 4)}s gpumem{gid}: ({mem_use} / {mem_total}) gpu_mem_usage{gid}: {mem_usage}%  gpu_compute_usage{gid}: {load}%')



def get_cpu_gpu_info():
  while True:
    get_cpu_info(time_start)
    get_all_gpu_info(occupy_mem, time_start)


def test():
  time_start = time.time()
  occupy_mem = [float(gpu.memoryUsed) for gpu in GPUtil.getGPUs()]
  get_all_gpu_info(occupy_mem, time_start)
  
  for i in range(10):
    get_cpu_info(time_start)
    get_all_gpu_info(occupy_mem, time_start)



if __name__ == '__main__':

  # test()

  time_start = time.time()
  time_start = 0
  while True:
    get_cpu_info(time_start)
    get_gpu_info(0, time_start)
