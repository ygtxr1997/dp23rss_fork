import torch
from accelerate import Accelerator


def main():
    # 1. 初始化 Accelerator (如果这里不报错，说明 NCCL 握手成功)
    accelerator = Accelerator()

    # 2. 获取当前进程的信息
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    print(
        f"[Rank {rank}] 成功启动！分配到的物理设备是: {device}, 当前显卡能看到的实际 GPU 数量: {torch.cuda.device_count()}")

    # 3. 在当前 GPU 上创建一个张量，值为 (rank + 1)
    tensor = torch.tensor([rank + 1.0]).to(device)

    # 等待所有 GPU 都运行到这里
    accelerator.wait_for_everyone()

    # 4. 强制多卡通信测试：把所有 GPU 上的 tensor 收集到一起
    try:
        gathered_tensor = accelerator.gather(tensor)

        if accelerator.is_main_process:
            print("\n" + "=" * 50)
            print("🚀 NCCL 多卡通信测试结果:")
            print(f"期望收集到 {world_size} 个值，实际收集结果: {gathered_tensor.cpu().numpy()}")
            print("如果上面输出了连续的数字，说明多卡分布式环境 100% 完美无缺！")
            print("=" * 50 + "\n")
    except Exception as e:
        print(f"[Rank {rank}] 通信失败，报错信息: {e}")


if __name__ == "__main__":
    main()