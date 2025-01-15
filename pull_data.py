import subprocess



for i in range(200) :
    subprocess.run(['wget', f'https://macaque.digital-brain.cn/cell/brain_region_T{i}_macaque_f001_Mq179-20230428.json'], shell='/bin/bash',executable=True)
