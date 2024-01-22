controlnet_mode = {
    'Depths': r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-depth-sdxl-1.0/',
    'Canny': r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-canny-sdxl-1.0/',
    'Lineart': r'/mnt/nfs/file_server/public/lipengxiang/sdxl_lineart',
}
a = r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-canny-sdxl-1.0/'
print(a in controlnet_mode.values())
key_name = next((key for key, value in controlnet_mode.items() if value == a), None)
print(key_name)
