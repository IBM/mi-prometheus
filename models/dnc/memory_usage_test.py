from memory_usage import MemoryUsage
#Tests for MemoryUsage
leng=10
mem_use=MemoryUsage(1)

usage=mem_use.init_state(10,3)
print(usage.shape)
print(usage)
usage[0,0]=20
usage[2,0]=2
usage[0,1:]=3
usage[2,2:]=4
usage[0,leng-1]=49494994
print(usage.shape)
print(usage)
cumprod=mem_use.exclusive_cumprod_temp(usage)
print(cumprod.shape)
print(cumprod)
