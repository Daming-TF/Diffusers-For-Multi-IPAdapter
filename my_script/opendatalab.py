import openxlab
AccessKey = '95ye0arz3ydlwggay7pw' 
SecretKey = 'kn63lvrz4m2z9e5pqkyvenaqzwq0roanodbpj8lb'
openxlab.login(ak=AccessKey, sk=SecretKey)

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/FFHQ')

from openxlab.dataset import query
query(dataset_repo='OpenDataLab/FFHQ')

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/FFHQ', target_path=r'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/FFHQ',source_path='/README.md', target_path=r'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data') #数据集文件下载pyth