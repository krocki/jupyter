# cuda cores per SM
cores_per_sm = {
    '-1' : -1,  # "Graphics Device"
    '3.0': 192, # Kepler
    '3.2': 192, # Kepler
    '3.5': 192, # Kepler
    '3.7': 192, # Kepler
    '5.0': 128, # Maxwell
    '5.2': 128, # Maxwell
    '5.3': 128, # Maxwell
    '6.0': 64,  # Pascal
    '6.1': 128, # Pascal
    '6.2': 128, # Pascal
    '7.0': 64,  # Volta
    '7.2': 64,  # Xavier
    '7.5': 64,  # Turing
    '8.0': 64,  # Ampere
    '8.6': 128  # Ampere
}

# Max CUDA cores ops per cycle per core
# FP64, FP32, FP16, INT8
fp_ops = {
    '7.0':  (1,    2, 8, 8),
    '7.5':  (1/16, 2, 8, 8),
    '8.0':  (1,    2, 8, 8),
    '8.6':  (1,    2, 8, 8)
}

# Max Tensor cores ops per cycle per SM
# FP64, FP32, FP16, INT8, INT4, INT1
tc_ops = {
    '7.0': (0,      0, 1024,    0,    0,     0),
    '7.5': (0,      0, 1024, 2048, 4096, 16384),
    '8.0': (128, 1024, 2048, 4096, 8192, 32768),
    '8.6': (128, 1024, 2048, 4096, 8192, 32768)
    # Ampere sparse (0, 2048, 4096, 8192, 16384, 0) ?
}

def query_dev(drv, dev_id):
  
    dev = drv.Device(dev_id)
    attributes = dev.get_attributes()
    sms = attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]
    mcr = attributes[drv.device_attribute.MEMORY_CLOCK_RATE]
    bus = attributes[drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
    l2s = attributes[drv.device_attribute.L2_CACHE_SIZE]
    clk = attributes[drv.device_attribute.CLOCK_RATE]
    cap = dev.compute_capability()
    arch = f'{cap[0]}.{cap[1]}'
    cores = cores_per_sm[arch] * sms

    print("Device #%d: %s" % (dev_id, dev.name()))
    print(f"  Arch: {arch}"
          f", Mem: {dev.total_memory()//(1<<20)} MB, {mcr*1000*bus*2*1e-9/8:.0f} GBps")

    print(f"  {sms} SMs, {cores} cores, {l2s / (1<<20)} MB L2"
          f", {clk * 1e-6} GHz")
    print(f"     TF/s:"
          f"    {fp_ops[arch][0] * clk * cores * 1e-9:6.2f} FP64"
          f"    {fp_ops[arch][1] * clk * cores * 1e-9:6.2f} FP32"
          f"    {fp_ops[arch][2] * clk * cores * 1e-9:6.2f} FP16")
    print(f"  TC TF/s:"
          f"    {tc_ops[arch][0] * clk * sms * 1e-9:6.2f} FP64"
          f"    {tc_ops[arch][1] * clk * sms * 1e-9:6.2f} FP32"
          f"    {tc_ops[arch][2] * clk * sms * 1e-9:6.2f} FP16")
    print()
    
def query_all(drv):
    print("%d device(s) found." % drv.Device.count())

    for ordinal in range(drv.Device.count()):
        query_dev(drv, ordinal)