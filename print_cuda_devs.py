#!/mnt/apps/aristotle/site/linux-centos7-x86_64/gcc-12.2.0/miniconda3-22.11.1-jophx4hvs7facqx66sootz7yi5zdswn3/bin/python
import torch

print('cuda availability:', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
print('name0:', torch.cuda.get_device_name(0))
print('name1:', torch.cuda.get_device_name(1))
