import paramiko
from tqdm import tqdm


def transfer_file(
        server_a_host, 
        server_a_username, 
        server_a_password, 
        server_a_filepath, 
        # server_b_host, 
        # server_b_username, 
        # server_b_password, 
        # server_b_directory
        save_dir,
    ):
    # 创建两个SSH客户端对象
    client_a = paramiko.SSHClient()
    client_b = paramiko.SSHClient()

    # 允许连接到未知主机
    client_a.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client_b.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 连接到服务器A
        client_a.connect(server_a_host, username=server_a_username, password=server_a_password)
        
        # 使用SFTP协议创建连接到服务器B
        with client_a.open_sftp() as sftp_a:
            # 从服务器A下载文件
            sftp_a.get(server_a_filepath, save_dir)

        # # 连接到服务器B
        # client_b.connect(server_b_host, username=server_b_username, password=server_b_password)

        # # 使用SFTP协议创建连接到服务器B
        # with client_b.open_sftp() as sftp_b:
        #     # 将文件上传到服务器B
        #     sftp_b.put('/tmp/temp_file', server_b_directory + '/filename')

        # 关闭连接
        client_a.close()
        # client_b.close()
        
        print("文件传输成功！")
    
    except Exception as e:
        print("文件传输失败:", str(e))
        raise


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_host", type=str, default="118.193.96.216")
    parser.add_argument("--server_username", type=str, default="sk")
    parser.add_argument("--server_password", type=str, default="Woaiseekoo2345!")
    parser.add_argument("--file_dir", type=str, default="/data_sk/lib/stylar_ai/stylar_prod/stable-diffusion-api/models")
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    lora_names = [
        'commercialPortrait_SDXLIP_v1.safetensors',
    ]
    embeds_names = [
        'commercialPortrait_SDXLIP_v1_ip_image_embeddings.pt'
    ]
    lora_paths = [os.path.join(args.file_dir, 'LyCORIS', name) for name in lora_names]
    embeds_paths = [os.path.join(args.file_dir, 'iplora', name) for name in embeds_names]
    file_paths = embeds_paths + lora_paths
    print(file_paths)

    for file_path in tqdm(file_paths):
        save_path = os.path.join(args.save_dir, os.path.basename(file_path))
        transfer_file(
            args.server_host, 
            args.server_username, 
            args.server_password, 
            file_path,
            # 'server_b_host', 
            # 'server_b_username', 
            # 'server_b_password', 
            save_path
        )
