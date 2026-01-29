import asyncio

from tapo import ApiClient


# ================= Config =================
tapo_email = "allenliu0416@163.com"
tapo_password = "LTX4947978"

# ================= Device =================

MY_DEVICES = {
    "zone1": "192.168.0.185",
    "zone2": "192.168.0.198",
    "zone3": "192.168.0.102",
    "zone4": "192.168.0.195",
    "zone5": "192.168.0.130",
    "zone6": "192.168.0.164",
    "zone7": "192.168.0.110",
}
# ===============================================


# 控制单个设备的函数
async def control_one_device(client, name, ip):
    print(f"🔄 [{name}] 正在连接 ({ip})...")

    try:
        # 连接设备
        device = await client.p100(ip)

        # 获取当前状态
        info = await device.get_device_info()

        # 开启开关
        if info.device_on:
            print(f"    [{name}] 当前开启 -> 正在关闭...")
            await device.off()
        else:
            print(f"    [{name}] 当前关闭 -> 正在开启...")
            await device.on()

    except Exception as e:
        print(f" [{name}] 控制失败: {e}")
        print("Check Account setting if occur 'User not found'，")


async def main():
    print(f"====== 准备控制 {len(MY_DEVICES)} 个设备 ======")

    # 1. 创建总客户端
    client = ApiClient(tapo_email, tapo_password)

    # 2. 创建任务列表
    tasks = []
    for name, ip in MY_DEVICES.items():
        # 把每个设备的控制任务加入列表
        task = control_one_device(client, name, ip)
        tasks.append(task)

    # 3. 等待所有任务完成
    await asyncio.gather(*tasks)

    print("====== ALL DONE ======")


if __name__ == "__main__":
    asyncio.run(main())
